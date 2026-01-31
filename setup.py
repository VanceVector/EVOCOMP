#!/usr/bin/env python3
"""Setup script for EVOCOMP package."""

from setuptools import setup, find_packages

setup(
    name="evocomp",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "z3-solver>=4.12.0",
        "redis>=4.5.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
        ],
    },
    python_requires=">=3.10",
)
