/*
 * TurboQuant Metal Shader Library
 *
 * Fused quantization/dequantization kernels for Apple Silicon (Metal 3+).
 * Designed for KV-cache compression in LLM inference.
 *
 * Kernels:
 *   - tq_mse_quantize_fused   : rotate + nearest-centroid + bit-pack (single pass)
 *   - tq_mse_dequantize_fused : unpack + codebook-lookup + rotate-back (single pass)
 *   - tq_prod_quantize_fused  : MSE quant + QJL sign(S·r) + pack (single pass)
 *   - tq_prod_dequantize_fused: unpack + MSE dequant + QJL residual (single pass)
 *   - tq_inner_product        : vectorised inner-product from compressed repr
 *
 * Storage layout (bit-packed):
 *   1-bit:  8 values per byte, packed MSB-first
 *   2-bit:  4 values per byte, packed MSB-first  [0xC0 0x30 0x0C 0x03]
 *   4-bit:  2 values per byte, packed MSB-first  [0xF0 0x0F]
 *   8-bit:  1 value  per byte (direct int8)
 *
 * Memory assumptions:
 *   - All matrices (rotation, S) are row-major float32
 *   - codebook is 1D float32 of length 2^b
 *   - Thread grid: one thread per (batch, head, token) triple
 *   - d (head_dim) is passed as constant; assumed <= 256
 */

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants & helpers
// ---------------------------------------------------------------------------

// Maximum head dimension supported in threadgroup memory
constant uint MAX_D = 256;

// Nearest centroid via binary search in sorted codebook
inline uint nearest_centroid(float val,
                              const device float* codebook,
                              uint n_codes) {
    // Binary search on midpoint boundaries
    uint lo = 0, hi = n_codes - 1;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        float boundary = (codebook[mid] + codebook[mid + 1]) * 0.5f;
        if (val <= boundary) hi = mid;
        else                  lo = mid + 1;
    }
    return lo;
}

// Pack 2-bit index into pre-cleared byte at correct position
inline void pack2_into(device uchar* out_bytes,
                       uint coord_idx,
                       uint code) {
    uint byte_idx = coord_idx >> 2;          // divide by 4
    uint shift    = 6 - ((coord_idx & 3) << 1); // 6,4,2,0
    // Atomic OR not needed: each thread owns a contiguous token's bytes
    out_bytes[byte_idx] |= (uchar)((code & 0x3) << shift);
}

inline void pack4_into(device uchar* out_bytes,
                       uint coord_idx,
                       uint code) {
    uint byte_idx = coord_idx >> 1;          // divide by 2
    uint shift    = (coord_idx & 1) ? 0 : 4;
    out_bytes[byte_idx] |= (uchar)((code & 0xF) << shift);
}

inline void pack1_into(device uchar* out_bytes,
                       uint coord_idx,
                       uint code) {
    uint byte_idx = coord_idx >> 3;          // divide by 8
    uint shift    = 7 - (coord_idx & 7);     // MSB first
    out_bytes[byte_idx] |= (uchar)((code & 0x1) << shift);
}

inline uint unpack2_from(const device uchar* in_bytes, uint coord_idx) {
    uint byte_idx = coord_idx >> 2;
    uint shift    = 6 - ((coord_idx & 3) << 1);
    return (in_bytes[byte_idx] >> shift) & 0x3;
}

inline uint unpack4_from(const device uchar* in_bytes, uint coord_idx) {
    uint byte_idx = coord_idx >> 1;
    uint shift    = (coord_idx & 1) ? 0 : 4;
    return (in_bytes[byte_idx] >> shift) & 0xF;
}

inline uint unpack1_from(const device uchar* in_bytes, uint coord_idx) {
    uint byte_idx = coord_idx >> 3;
    uint shift    = 7 - (coord_idx & 7);
    return (in_bytes[byte_idx] >> shift) & 0x1;
}

// Sign-safe int8 → float (QJL: stored as 0=negative, 1=positive)
inline float qjl_bit_to_float(uint bit) {
    return (bit == 1) ? 1.0f : -1.0f;
}

// ---------------------------------------------------------------------------
// MSE Quantize: rotate + nearest-centroid + 2-bit pack
// Supports b in {1, 2, 4, 8}
//
// Grid: (n_tokens,) threads — one thread handles one full d-vector
// ---------------------------------------------------------------------------

kernel void tq_mse_quantize_fused(
    const device float*  x            [[ buffer(0) ]],   // (N, d) input
    const device float*  rotation     [[ buffer(1) ]],   // (d, d) rotation matrix
    const device float*  codebook     [[ buffer(2) ]],   // (2^b,) centroids
          device uchar*  packed_out   [[ buffer(3) ]],   // (N, ceil(b*d/8)) packed bytes
          device float*  norms_out    [[ buffer(4) ]],   // (N,) optional norms (float32)
    constant uint&       d            [[ buffer(5) ]],
    constant uint&       b            [[ buffer(6) ]],
    constant uint&       normalize    [[ buffer(7) ]],   // 1=normalize input, 0=use as-is
    uint                 token_idx    [[ thread_position_in_grid ]]
) {
    uint n_codes    = 1u << b;
    uint bytes_per  = (b * d + 7) / 8;          // packed bytes per token

    // Load input vector
    threadgroup float y[MAX_D];
    float norm = 0.0f;
    for (uint j = 0; j < d; ++j) {
        float v = x[token_idx * d + j];
        y[j] = v;
        norm += v * v;
    }
    norm = sqrt(norm);
    if (norm < 1e-12f) norm = 1.0f;

    if (normalize) {
        norms_out[token_idx] = norm;
        for (uint j = 0; j < d; ++j) y[j] /= norm;
    }

    // Rotate: z = rotation @ y  (rotation is row-major: rotation[i*d + j])
    threadgroup float z[MAX_D];
    for (uint i = 0; i < d; ++i) {
        float acc = 0.0f;
        for (uint j = 0; j < d; ++j) acc += rotation[i * d + j] * y[j];
        z[i] = acc;
    }

    // Nearest centroid + pack
    device uchar* out_ptr = packed_out + token_idx * bytes_per;

    // Zero out the output bytes for this token first
    for (uint byte = 0; byte < bytes_per; ++byte) out_ptr[byte] = 0;

    for (uint i = 0; i < d; ++i) {
        uint code = nearest_centroid(z[i], codebook, n_codes);
        if (b == 1) {
            pack1_into(out_ptr, i, code);
        } else if (b == 2) {
            pack2_into(out_ptr, i, code);
        } else if (b == 4) {
            pack4_into(out_ptr, i, code);
        } else {
            // b == 8: direct byte store
            out_ptr[i] = (uchar)(code & 0xFF);
        }
    }
}

// ---------------------------------------------------------------------------
// MSE Dequantize: unpack + codebook-lookup + rotate-back
// ---------------------------------------------------------------------------

kernel void tq_mse_dequantize_fused(
    const device uchar*  packed_in    [[ buffer(0) ]],   // (N, ceil(b*d/8))
    const device float*  rotation     [[ buffer(1) ]],   // (d, d) — we apply rotationᵀ
    const device float*  codebook     [[ buffer(2) ]],   // (2^b,)
    const device float*  norms_in     [[ buffer(3) ]],   // (N,) or nullptr if unnormalized
          device float*  x_out        [[ buffer(4) ]],   // (N, d) output
    constant uint&       d            [[ buffer(5) ]],
    constant uint&       b            [[ buffer(6) ]],
    constant uint&       has_norm     [[ buffer(7) ]],   // 1=scale by norm, 0=skip
    uint                 token_idx    [[ thread_position_in_grid ]]
) {
    uint bytes_per = (b * d + 7) / 8;

    const device uchar* in_ptr = packed_in + token_idx * bytes_per;

    // Unpack → codebook values
    threadgroup float y[MAX_D];
    for (uint i = 0; i < d; ++i) {
        uint code;
        if (b == 1)      code = unpack1_from(in_ptr, i);
        else if (b == 2) code = unpack2_from(in_ptr, i);
        else if (b == 4) code = unpack4_from(in_ptr, i);
        else             code = (uint)in_ptr[i];
        y[i] = codebook[code];
    }

    // Rotate back: x = rotationᵀ @ y  (= rotation[j*d + i] for column i)
    float norm = has_norm ? norms_in[token_idx] : 1.0f;
    for (uint i = 0; i < d; ++i) {
        float acc = 0.0f;
        for (uint j = 0; j < d; ++j) acc += rotation[j * d + i] * y[j];
        x_out[token_idx * d + i] = acc * norm;
    }
}

// ---------------------------------------------------------------------------
// Prod Quantize: MSE(b-1 bits) + QJL on residual
//
// Output:
//   packed_idx  : (N, ceil((b-1)*d/8)) packed MSE indices
//   packed_qjl  : (N, ceil(d/8))       packed QJL bits (1=positive, 0=negative)
//   gamma_out   : (N,)                 residual L2 norm
// ---------------------------------------------------------------------------

kernel void tq_prod_quantize_fused(
    const device float*  x            [[ buffer(0) ]],   // (N, d)
    const device float*  rotation     [[ buffer(1) ]],   // (d, d) MSE rotation
    const device float*  codebook     [[ buffer(2) ]],   // (2^(b-1),) MSE codebook
    const device float*  S            [[ buffer(3) ]],   // (d, d) QJL projection
          device uchar*  packed_idx   [[ buffer(4) ]],   // (N, ceil((b-1)*d/8))
          device uchar*  packed_qjl   [[ buffer(5) ]],   // (N, ceil(d/8))
          device float*  gamma_out    [[ buffer(6) ]],   // (N,)
    constant uint&       d            [[ buffer(7) ]],
    constant uint&       b            [[ buffer(8) ]],   // total bits (b >= 1)
    uint                 token_idx    [[ thread_position_in_grid ]]
) {
    uint b_mse  = (b <= 1) ? 1u : (b - 1);
    uint n_codes = 1u << b_mse;
    uint idx_bytes = (b_mse * d + 7) / 8;
    uint qjl_bytes = (d + 7) / 8;

    device uchar* idx_ptr = packed_idx + token_idx * idx_bytes;
    device uchar* qjl_ptr = packed_qjl + token_idx * qjl_bytes;

    for (uint byte = 0; byte < idx_bytes; ++byte) idx_ptr[byte] = 0;
    for (uint byte = 0; byte < qjl_bytes; ++byte) qjl_ptr[byte] = 0;

    // Load x
    threadgroup float xv[MAX_D];
    for (uint j = 0; j < d; ++j) xv[j] = x[token_idx * d + j];

    if (b <= 1) {
        // Skip MSE stage: residual = x, idx all zeros (already cleared above)
        // QJL: sign(S @ x)
        float gamma_sq = 0.0f;
        for (uint j = 0; j < d; ++j) gamma_sq += xv[j] * xv[j];
        gamma_out[token_idx] = sqrt(gamma_sq);

        for (uint i = 0; i < d; ++i) {
            float acc = 0.0f;
            for (uint j = 0; j < d; ++j) acc += S[i * d + j] * xv[j];
            pack1_into(qjl_ptr, i, acc >= 0.0f ? 1u : 0u);
        }
        return;
    }

    // Step 1: rotate x → z
    threadgroup float z[MAX_D];
    for (uint i = 0; i < d; ++i) {
        float acc = 0.0f;
        for (uint j = 0; j < d; ++j) acc += rotation[i * d + j] * xv[j];
        z[i] = acc;
    }

    // Step 2: nearest centroid + reconstruct x̃_mse
    threadgroup float x_mse[MAX_D];
    for (uint i = 0; i < d; ++i) {
        uint code = nearest_centroid(z[i], codebook, n_codes);
        if (b_mse == 1)      pack1_into(idx_ptr, i, code);
        else if (b_mse == 2) pack2_into(idx_ptr, i, code);
        else if (b_mse == 4) pack4_into(idx_ptr, i, code);
        else                 idx_ptr[i] = (uchar)(code & 0xFF);
        z[i] = codebook[code];  // reuse z to store ỹ
    }

    // Rotate back: x̃_mse = rotationᵀ @ ỹ
    for (uint i = 0; i < d; ++i) {
        float acc = 0.0f;
        for (uint j = 0; j < d; ++j) acc += rotation[j * d + i] * z[j];
        x_mse[i] = acc;
    }

    // Step 3: residual r = x - x̃_mse, compute ||r||, QJL
    threadgroup float r[MAX_D];
    float gamma_sq = 0.0f;
    for (uint j = 0; j < d; ++j) {
        r[j] = xv[j] - x_mse[j];
        gamma_sq += r[j] * r[j];
    }
    gamma_out[token_idx] = sqrt(gamma_sq);

    // sign(S @ r) — S is row-major, S[i*d + j]
    for (uint i = 0; i < d; ++i) {
        float acc = 0.0f;
        for (uint j = 0; j < d; ++j) acc += S[i * d + j] * r[j];
        pack1_into(qjl_ptr, i, acc >= 0.0f ? 1u : 0u);
    }
}

// ---------------------------------------------------------------------------
// Prod Dequantize: unpack MSE + unpack QJL + combine
// ---------------------------------------------------------------------------

kernel void tq_prod_dequantize_fused(
    const device uchar*  packed_idx   [[ buffer(0) ]],   // (N, ceil((b-1)*d/8))
    const device uchar*  packed_qjl   [[ buffer(1) ]],   // (N, ceil(d/8))
    const device float*  gamma_in     [[ buffer(2) ]],   // (N,)
    const device float*  rotation     [[ buffer(3) ]],   // (d, d)
    const device float*  codebook     [[ buffer(4) ]],   // (2^(b-1),)
    const device float*  S            [[ buffer(5) ]],   // (d, d)
          device float*  x_out        [[ buffer(6) ]],   // (N, d)
    constant uint&       d            [[ buffer(7) ]],
    constant uint&       b            [[ buffer(8) ]],
    uint                 token_idx    [[ thread_position_in_grid ]]
) {
    uint b_mse   = (b <= 1) ? 1u : (b - 1);
    uint idx_bytes = (b_mse * d + 7) / 8;
    uint qjl_bytes = (d + 7) / 8;
    float qjl_scale = sqrt(M_PI_F / 2.0f) / (float)d;

    const device uchar* idx_ptr = packed_idx + token_idx * idx_bytes;
    const device uchar* qjl_ptr = packed_qjl + token_idx * qjl_bytes;
    float gamma = gamma_in[token_idx];

    if (b <= 1) {
        // x̃_mse = 0; x̃_qjl = scale * gamma * Sᵀ @ qjl
        for (uint i = 0; i < d; ++i) {
            float acc = 0.0f;
            for (uint j = 0; j < d; ++j) {
                float qjl_val = qjl_bit_to_float(unpack1_from(qjl_ptr, j));
                acc += S[j * d + i] * qjl_val;  // Sᵀ[i,j] = S[j,i]
            }
            x_out[token_idx * d + i] = qjl_scale * gamma * acc;
        }
        return;
    }

    // Unpack MSE indices → codebook → rotate back
    threadgroup float y_mse[MAX_D];
    for (uint i = 0; i < d; ++i) {
        uint code;
        if (b_mse == 1)      code = unpack1_from(idx_ptr, i);
        else if (b_mse == 2) code = unpack2_from(idx_ptr, i);
        else if (b_mse == 4) code = unpack4_from(idx_ptr, i);
        else                 code = (uint)idx_ptr[i];
        y_mse[i] = codebook[code];
    }

    // x̃_mse = rotationᵀ @ y_mse
    threadgroup float x_mse[MAX_D];
    for (uint i = 0; i < d; ++i) {
        float acc = 0.0f;
        for (uint j = 0; j < d; ++j) acc += rotation[j * d + i] * y_mse[j];
        x_mse[i] = acc;
    }

    // x̃_qjl = scale * gamma * Sᵀ @ qjl
    // Combined output
    for (uint i = 0; i < d; ++i) {
        float acc_qjl = 0.0f;
        for (uint j = 0; j < d; ++j) {
            float qjl_val = qjl_bit_to_float(unpack1_from(qjl_ptr, j));
            acc_qjl += S[j * d + i] * qjl_val;
        }
        x_out[token_idx * d + i] = x_mse[i] + qjl_scale * gamma * acc_qjl;
    }
}

// ---------------------------------------------------------------------------
// Inner-product estimation (fast path for ANN search)
//
// Computes <y, x̃> for each x̃ given compressed (idx, qjl, gamma)
// without fully reconstructing x̃.
//
// Grid: (n_tokens,) threads
// ---------------------------------------------------------------------------

kernel void tq_inner_product(
    const device float*  query        [[ buffer(0) ]],   // (d,) query vector
    const device uchar*  packed_idx   [[ buffer(1) ]],   // (N, ceil((b-1)*d/8))
    const device uchar*  packed_qjl   [[ buffer(2) ]],   // (N, ceil(d/8))
    const device float*  gamma_in     [[ buffer(3) ]],   // (N,)
    const device float*  rotation     [[ buffer(4) ]],   // (d, d)
    const device float*  codebook     [[ buffer(5) ]],   // (2^(b-1),)
    const device float*  S            [[ buffer(6) ]],   // (d, d)
          device float*  ip_out       [[ buffer(7) ]],   // (N,) inner products
    constant uint&       d            [[ buffer(8) ]],
    constant uint&       b            [[ buffer(9) ]],
    uint                 token_idx    [[ thread_position_in_grid ]]
) {
    uint b_mse   = (b <= 1) ? 1u : (b - 1);
    uint idx_bytes = (b_mse * d + 7) / 8;
    uint qjl_bytes = (d + 7) / 8;
    float qjl_scale = sqrt(M_PI_F / 2.0f) / (float)d;

    const device uchar* idx_ptr = packed_idx + token_idx * idx_bytes;
    const device uchar* qjl_ptr = packed_qjl + token_idx * qjl_bytes;
    float gamma = gamma_in[token_idx];

    float ip_mse = 0.0f;
    if (b > 1) {
        // x̃_mse = rotationᵀ @ codebook[idx]
        // <y, x̃_mse> = <rotation @ y, codebook[idx]> (rotation is orthogonal)
        // Compute ry = rotation @ y first
        threadgroup float ry[MAX_D];
        for (uint i = 0; i < d; ++i) {
            float acc = 0.0f;
            for (uint j = 0; j < d; ++j) acc += rotation[i * d + j] * query[j];
            ry[i] = acc;
        }
        for (uint i = 0; i < d; ++i) {
            uint code;
            if (b_mse == 1)      code = unpack1_from(idx_ptr, i);
            else if (b_mse == 2) code = unpack2_from(idx_ptr, i);
            else if (b_mse == 4) code = unpack4_from(idx_ptr, i);
            else                 code = (uint)idx_ptr[i];
            ip_mse += ry[i] * codebook[code];
        }
    }

    // ip_qjl = scale * gamma * <S @ y, qjl>
    float ip_qjl = 0.0f;
    for (uint i = 0; i < d; ++i) {
        float sy_i = 0.0f;
        for (uint j = 0; j < d; ++j) sy_i += S[i * d + j] * query[j];
        float qjl_val = qjl_bit_to_float(unpack1_from(qjl_ptr, i));
        ip_qjl += sy_i * qjl_val;
    }
    ip_qjl *= qjl_scale * gamma;

    ip_out[token_idx] = ip_mse + ip_qjl;
}

// ---------------------------------------------------------------------------
// Batch-normalise vectors before quantization (fused norm extraction)
// Grid: (N,) — each thread normalises one vector
// ---------------------------------------------------------------------------

kernel void tq_normalize_vectors(
    const device float*  x_in     [[ buffer(0) ]],   // (N, d)
          device float*  x_out    [[ buffer(1) ]],   // (N, d)
          device float*  norms    [[ buffer(2) ]],   // (N,)
    constant uint&       d        [[ buffer(3) ]],
    uint                 token_idx [[ thread_position_in_grid ]]
) {
    float norm_sq = 0.0f;
    for (uint j = 0; j < d; ++j) {
        float v = x_in[token_idx * d + j];
        norm_sq += v * v;
    }
    float norm = sqrt(norm_sq);
    if (norm < 1e-12f) norm = 1.0f;
    norms[token_idx] = norm;
    for (uint j = 0; j < d; ++j) {
        x_out[token_idx * d + j] = x_in[token_idx * d + j] / norm;
    }
}
