from setuptools import setup, Extension
import Cython.Build
import pip
pip.main(["install", "torch"])

setup(
        ext_modules = [
                Extension(
                        "AnalysisTopGNN.Vectors", 
                        sources = ["src/Vectors/Lorentz.pyx"]
                )
        ],
        cmdclass = {"build_ext" : Cython.Build.build_ext}
)
