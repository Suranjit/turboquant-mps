from setuptools import setup, find_packages

setup(
    name="turboquant",
    version="0.1.0",
    description="TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "torch>=2.1",
    ],
    extras_require={
        "llm": ["transformers>=4.38", "accelerate>=0.26", "matplotlib>=3.7"],
    },
)
