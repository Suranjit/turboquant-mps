"""
Metal library compiler and runtime loader.

Compiles the MSL source (kernels.metal) into a Metal library using the
system `xcrun metal` + `xcrun metallib` toolchain, then loads it via
PyObjC's Metal framework bindings.

Falls back gracefully: if PyObjC is unavailable or compilation fails,
MetalTurboQuantMSE/Prod will fall through to the PyTorch-MPS path.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path to the MSL source bundled with this package
_KERNEL_SRC = Path(__file__).parent / "kernels.metal"

# Persistent compiled library cache (next to the .metal source)
_LIB_CACHE  = Path(__file__).parent / "_compiled_turboquant.metallib"


def _sdk_path() -> Optional[str]:
    """Return the macOS SDK path, or None if not on macOS."""
    try:
        result = subprocess.run(
            ["xcrun", "--show-sdk-path"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _source_hash() -> str:
    """SHA-256 of the .metal source, used to invalidate the cache."""
    return hashlib.sha256(_KERNEL_SRC.read_bytes()).hexdigest()[:16]


def _hash_file() -> Path:
    return _LIB_CACHE.with_suffix(".hash")


def compile_metal_lib(force: bool = False) -> Optional[Path]:
    """
    Compile kernels.metal → .metallib using xcrun toolchain.

    Returns the path to the compiled .metallib, or None on failure.
    """
    if not _KERNEL_SRC.exists():
        logger.error("Metal kernel source not found: %s", _KERNEL_SRC)
        return None

    sdk = _sdk_path()
    if sdk is None:
        logger.warning("xcrun not available — Metal compilation skipped")
        return None

    # Check if cached library is up-to-date  # pragma: no cover
    src_hash = _source_hash()  # pragma: no cover
    if (not force  # pragma: no cover
            and _LIB_CACHE.exists()
            and _hash_file().exists()
            and _hash_file().read_text().strip() == src_hash):
        logger.debug("Using cached Metal library: %s", _LIB_CACHE)  # pragma: no cover
        return _LIB_CACHE  # pragma: no cover

    logger.info("Compiling Metal shaders → %s ...", _LIB_CACHE)  # pragma: no cover

    with tempfile.TemporaryDirectory() as tmp:  # pragma: no cover
        air = Path(tmp) / "kernels.air"

        # Step 1: .metal → .air (intermediate representation)
        ret = subprocess.run([
            "xcrun", "metal",
            "-sdk", "macosx",
            "-target", "air64-apple-macosx14.0",
            "-O2",
            "-Wall",
            "-c", str(_KERNEL_SRC),
            "-o", str(air),
        ], capture_output=True, text=True)

        if ret.returncode != 0:
            logger.error("Metal compilation failed:\n%s\n%s", ret.stdout, ret.stderr)
            return None

        # Step 2: .air → .metallib
        ret = subprocess.run([
            "xcrun", "metallib",
            str(air),
            "-o", str(_LIB_CACHE),
        ], capture_output=True, text=True)

        if ret.returncode != 0:
            logger.error("metallib linking failed:\n%s\n%s", ret.stdout, ret.stderr)
            return None

    _hash_file().write_text(src_hash)
    logger.info("Metal library compiled successfully: %s", _LIB_CACHE)
    return _LIB_CACHE


def metal_available() -> bool:
    """True if Metal is available and PyObjC Metal bindings are importable."""
    try:
        import Metal  # noqa: F401  (PyObjC)
        return True
    except ImportError:
        return False


class MetalLib:  # pragma: no cover
    """
    Thin wrapper around a compiled Metal library + command queue.

    Usage::

        lib = MetalLib.load()
        if lib is None:
            # fall back to torch MPS
            ...

        buf = lib.make_buffer(tensor)
        lib.run_kernel("tq_mse_quantize_fused", buffers=[...], grid=N)
    """

    def __init__(self, mtl_device, mtl_library, command_queue):
        self._device  = mtl_device
        self._library = mtl_library
        self._queue   = command_queue
        self._pipelines: dict[str, object] = {}

    # ------------------------------------------------------------------

    @classmethod
    def load(cls, force_recompile: bool = False) -> Optional["MetalLib"]:
        """
        Compile (if needed) and load the Metal library.

        Returns None if Metal/PyObjC is unavailable or compilation fails.
        """
        if not metal_available():
            logger.info("PyObjC Metal not available; using PyTorch MPS fallback")
            return None

        lib_path = compile_metal_lib(force=force_recompile)
        if lib_path is None:
            return None

        try:
            import Metal
            import objc  # noqa: F401

            device = Metal.MTLCreateSystemDefaultDevice()
            if device is None:
                logger.warning("No Metal device found")
                return None

            lib_url = objc.NSURL.fileURLWithPath_(str(lib_path))
            mtl_lib, err = device.newLibraryWithURL_error_(lib_url, None)
            if err is not None:
                logger.error("Failed to load .metallib: %s", err)
                return None

            queue = device.newCommandQueue()
            return cls(device, mtl_lib, queue)

        except Exception as exc:
            logger.warning("MetalLib.load() failed: %s", exc)
            return None

    # ------------------------------------------------------------------

    def _get_pipeline(self, kernel_name: str):
        """Cache compute pipeline state for a kernel function."""
        if kernel_name in self._pipelines:
            return self._pipelines[kernel_name]

        import Metal
        fn = self._library.newFunctionWithName_(kernel_name)
        if fn is None:
            raise ValueError(f"Metal kernel '{kernel_name}' not found in library")
        pipeline, err = self._device.newComputePipelineStateWithFunction_error_(fn, None)
        if err is not None:
            raise RuntimeError(f"Pipeline creation failed for '{kernel_name}': {err}")
        self._pipelines[kernel_name] = pipeline
        return pipeline

    def make_buffer_from_numpy(self, arr) -> object:
        """Create a shared Metal buffer from a numpy array (zero-copy on Apple Silicon)."""
        import numpy as np
        import Metal
        arr = np.ascontiguousarray(arr)
        nbytes = arr.nbytes
        # MTLResourceStorageModeShared = 0
        buf = self._device.newBufferWithBytes_length_options_(
            arr.ctypes.data, nbytes, 0
        )
        return buf

    def make_buffer_from_torch(self, tensor) -> object:
        """
        Create a Metal buffer backed by a contiguous torch tensor.
        For MPS tensors we use the data_ptr directly via ctypes.
        """
        import ctypes
        import Metal
        tensor = tensor.contiguous()
        nbytes = tensor.element_size() * tensor.numel()
        # For CPU tensors: zero-copy via data_ptr
        if tensor.device.type == "cpu":
            buf = self._device.newBufferWithBytes_length_options_(
                ctypes.c_void_p(tensor.data_ptr()), nbytes, 0
            )
        else:
            # MPS tensor: copy to CPU first (still on unified memory — fast)
            cpu_t = tensor.cpu().contiguous()
            buf = self._device.newBufferWithBytes_length_options_(
                ctypes.c_void_p(cpu_t.data_ptr()), nbytes, 0
            )
        return buf

    def make_output_buffer(self, nbytes: int) -> object:
        """Allocate a Metal output buffer (MTLResourceStorageModeShared)."""
        import Metal
        return self._device.newBufferWithLength_options_(nbytes, 0)

    def run_kernel(
        self,
        kernel_name: str,
        buffers: list,
        grid: int,
        threadgroup_size: int = 1,
    ) -> None:
        """
        Dispatch a 1D compute kernel synchronously.

        Args:
            kernel_name:      Name of the MSL kernel function.
            buffers:          List of MTLBuffer objects (in binding order).
            grid:             Total number of threads to dispatch.
            threadgroup_size: Threads per threadgroup (default 1 for token-level kernels).
        """
        import Metal

        pipeline = self._get_pipeline(kernel_name)
        cmd_buf  = self._queue.commandBuffer()
        encoder  = cmd_buf.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)

        tg_size = Metal.MTLSizeMake(threadgroup_size, 1, 1)
        grid_sz = Metal.MTLSizeMake(grid, 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid_sz, tg_size)
        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

    def buffer_to_numpy(self, buf, shape, dtype) -> "np.ndarray":
        """Copy a Metal buffer back to a numpy array."""
        import numpy as np
        import ctypes
        ptr = buf.contents()
        nbytes = buf.length()
        arr = np.frombuffer(
            (ctypes.c_byte * nbytes).from_address(ctypes.addressof(ptr)), dtype=dtype
        ).reshape(shape).copy()
        return arr
