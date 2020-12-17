import subprocess
import platform
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

with open("README.md", "r") as fh:
    long_description = fh.read()


def install_graphviz() -> None:
    if str(platform.system()) == "Darwin":
        subprocess.call("brew install graphviz", shell=True)
    elif str(platform.system()) == "Linux":
        subprocess.call(
            "sudo apt -y install build-essential python-dev libxml2 libxml2-dev zlib1g-dev",
            shell=True,
        )
        subprocess.call("sudo apt -y install python-pydot python-pydot-ng graphviz", shell=True)
    else:
        raise NotImplementedError(
            f"We're really sorry, but {platform.system()} isn't a supported OS"
        )


class CustomInstallCommand(install):
    """Custom install setup to help run shell commands (outside shell) before installation"""

    def run(self) -> None:
        install.run(self)
        install_graphviz()


class CustomDevelopCommand(develop):
    """Custom install setup to help run shell commands (outside shell) before installation of dev"""

    def run(self) -> None:
        develop.run(self)
        install_graphviz()


setup(
    cmdclass={"install": CustomInstallCommand, "develop": CustomDevelopCommand},
    name="BayNet",
    version="0.2.2",
    author="Chris Robinson",
    author_email="c.f.robinson@sussex.ac.uk",
    description="(another) Python Bayesian Network library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Stoffle/BayNet",
    packages=find_packages(exclude=("tests",)),
    package_data={'baynet': ['baynet/utils/bif_library/*.bif']},
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "python-igraph < 0.8.0",
        "numpy >= 1.17.2",
        "pandas >= 0.25",
        "protobuf",
        "graphviz",
        "pyparsing",
    ],
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
