#!/usr/bin/env python3
"""Setup script for PDF2MD - PDF to Markdown Converter."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="pdf2md",
    version="1.0.0",
    description="Convert PDF documents to Markdown using IBM Docling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="PDF2MD",
    author_email="",
    url="https://github.com/yourusername/pdf2md",
    py_modules=["pdf2md"],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pdf2md=pdf2md:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Text Processing",
        "Topic :: Text Processing :: Markup",
        "Topic :: Utilities",
    ],
    keywords="pdf markdown converter docling cli",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pdf2md/issues",
        "Source": "https://github.com/yourusername/pdf2md",
    },
)