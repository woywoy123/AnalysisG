from setuptools import build_meta as _orig
from setuptools.build_meta import *
import subprocess
import os

def _getTorch():
    def _getcmd(cmd):
        return subprocess.check_output(cmd, stderr = subprocess.STDOUT, shell = True).decode("UTF-8")

    gcc_cu = {
                        12  : ["12.1.0"], 
                        11  : ["11.4.1", "11.5", "11.6", "11.7", "11.8"], 
                        10  : ["11.1", "11.2", "11.3", "11.4.0"], 
                        9   : ["11"], 
                        8   : ["10.1", "10.2"], 
                        7   : ["9.2", "10.0"], 
                        6   : ["9.0", "9.1"], 
                        5.3 : ["8"], 
    }

    nvcc = _getcmd("n --version | grep release | awk '{print $5}'")[:-2]
    this_gcc = [i for i in gcc_cu if nvcc in gcc_cu[i]]
    gcc = _getcmd("gcc --version | grep gcc | awk '{print $3}'").split("-")[0]

    if len(this_gcc) > 0 and float(this_gcc[0]) <= this_gcc[0]: return "cu" + nvcc.replace(".", "")
    f = open("cpu.txt", "w")
    f.write("")
    f.close()

def get_requires_for_build_wheel(self, config_settings = None):
    _getTorch() 
    return _orig.get_requires_for_build_wheel(config_settings)

def get_requires_for_build_sdist(self, config_settings = None):
    _getTorch() 
    return orig.get_requires_for_build_sdist(config_settings)

def get_requires_for_build_editable(self, config_settings = None):
    _getTorch()
    return _orig.get_requires_for_build_editable(config_settings)
