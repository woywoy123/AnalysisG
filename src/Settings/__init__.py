from .Settings import Settings
import subprocess
from subprocess import Popen, PIPE, STDOUT
from pwinput import pwinput

def _getcmd(cmd):

    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode(
        "UTF-8"
    )


def CONFIG_PYAMI():
    print("------- CONFIGURING PYAMI -------")
    print(_getcmd('echo "y" | ami_atlas_post_install'))
    print("-------- FINISHED PYAMI ---------")

def AUTH_PYAMI():
    print("Please specify the directory where your .globus directory is located.")
    globu = input("(default: ~/.globus): ")
    globu = "~/.globus/" if globu == "" else globu
    p = Popen(
        ["voms-proxy-init", "-certdir", globu, "-verify"],
        stdout = PIPE, stdin = PIPE, stderr = STDOUT
    )

    print("Provide the password to decrypt the PEM files. Leave blank and press enter for no password.")
    code = pwinput("Password: ")
    stdout = p.communicate(code.encode("UTF-8"))
    print(stdout)

def POST_INSTALL_PYC():
    def stdout():
        for line in p.stdout:
            print(line.decode("UTF-8").replace("\n", ""))

    import subprocess
    import sys
    from threading import Thread
    import pathlib

    print("------- CONFIGURING TORCH-EXTENSIONS (PYC) -------")
    p = subprocess.Popen(
        [sys.executable, "-m", "pip", "install", "-v", "./torch-extensions/"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    t = Thread(target=stdout, daemon=True)
    t.start()
    t.join()
    print("------- DONE CONFIGURING TORCH-EXTENSIONS (PYC) -------")

