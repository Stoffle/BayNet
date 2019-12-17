import platform
import subprocess
from pathlib import Path
from typing import List, Dict

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


with open("README.md", "r") as fh:
    long_description = fh.read()

PY_V = platform.python_version().replace(".", "")[:2]


setup(
    name="BayNet",
    version="0.0.1-dev",
    author="Chris Robinson",
    author_email="c.f.robinson@sussex.ac.uk",
    description="(another) Python Bayesian Network library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Stoffle/BayNet",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=["python-igraph >= 0.7.0", "numpy >= 1.17.2", "pandas >= 0.25",],
    extras_require={
        "dev": [
            "black",
            "mypy >= 0.720",
            "pylint >= 2.0",
            "pytest >= 3.3.2",
            "pytest-cov >= 2.6.0",
            "pre-commit",
            "pydocstyle",
        ],
        "ci": ["pytest >= 3.3.2", "pytest-cov >= 2.6.0"],
    },
)
