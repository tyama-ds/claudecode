#!/usr/bin/env python3
"""
Setup script for Deep Research Tool.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "deep_research_tool" / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="deep-research-tool",
    version="0.1.0",
    author="Deep Research Tool Team",
    author_email="",
    description="Automated research tool with AI assistance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/deep-research-tool",
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "black>=24.0.0",
            "isort>=5.13.0",
            "mypy>=1.8.0",
        ],
        "selenium": [
            "selenium>=4.20.0",
            "webdriver-manager>=4.0.0",
        ],
        "units": [
            "pint>=0.24.0",  # Optional: dimensional analysis (Level 3 unit conversion)
        ],
    },
    entry_points={
        "console_scripts": [
            "deep-research=deep_research_tool.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
