"""
Setup configuration for LSP2Graph
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lsp2graph",
    version="0.1.0",
    author="LSP2Graph Team",
    author_email="team@lsp2graph.dev",
    description="Universal Code Extraction System using LSP and AST analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/lsp2graph",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",
        "tree-sitter>=0.20.0",
        "tree-sitter-python>=0.20.0",
        "tree-sitter-javascript>=0.20.0",
        "tree-sitter-typescript>=0.20.0",
        "tree-sitter-rust>=0.20.0",
        "tree-sitter-go>=0.20.0",
        "tree-sitter-cpp>=0.20.0",
        "tree-sitter-c>=0.20.0",
        "tree-sitter-java>=0.20.0",
        "tree-sitter-haskell>=0.20.0",
        "tree-sitter-ruby>=0.20.0",
        "tree-sitter-php>=0.20.0",
        "tree-sitter-bash>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [
            "lsp2graph-extract=lsp2graph.extractors.ast_extractor:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
