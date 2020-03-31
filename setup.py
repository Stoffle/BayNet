from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="BayNet",
    version="0.1.2",
    author="Chris Robinson",
    author_email="c.f.robinson@sussex.ac.uk",
    description="(another) Python Bayesian Network library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Stoffle/BayNet",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
    install_requires=["python-igraph < 0.8.0", "numpy >= 1.17.2", "pandas >= 0.25", "pyyaml"],
    extras_require={
        "dev": [
            "black",
            "mypy >= 0.720",
            "pylint >= 2.0",
            "pytest >= 3.3.2",
            "pytest-cov >= 2.6.0",
            "pre-commit",
            "pydocstyle",
            "networkx",
        ],
        "ci": ["pytest >= 3.3.2", "pytest-cov >= 2.6.0", "networkx"],
    },
)
