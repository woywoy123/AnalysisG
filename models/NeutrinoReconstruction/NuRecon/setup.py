from setuptools import setup, find_namespace_packages
from setuptools.command.install import install
import subprocess

class LocalInstall(install):
    def run(self):
        install.run(self)
        subprocess.call("pip install ./Physics/Floats/", shell=True)
        subprocess.call("pip install ./Physics/Tensors/", shell=True)

setuptools.setup(
        name = "NuSolutions", 
        packages = find_namespace_packages(include = ["Physics.*"]),
        cmdclass = { "install": LocalInstall },
    )
