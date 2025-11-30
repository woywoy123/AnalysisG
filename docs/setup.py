from setuptools import setup, find_packages

setup(
    name="docs",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "sphinx",
        "breathe",
        "sphinx_rtd_theme",
    ],
)