from pwinput import pwinput
import subprocess

global init_ami

def _getcmd(cmd):
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode("UTF-8")

def config_pyami():
    print("------- CONFIGURING PYAMI -------")
    print(_getcmd('echo "y" | ami_atlas_post_install'))
    print("-------- FINISHED PYAMI ---------")

def auth_pyami():
    if init_ami is not None: return
    init_ami = True

    print("Please specify the directory where your .globus directory is located.")
    globu = input("(default: ~/.globus): ")
    globu = "~/.globus/" if globu == "" else globu
    print("Provide the password to decrypt the PEM files. Leave blank and press enter for no password.")
    execute = " ".join(["voms-proxy-init", "-certdir", globu, "-pwstdin"])
    code = pwinput("Password: ")
    execute = 'echo "' + code + '" | ' + execute
    try: _getcmd(execute)
    except subprocess.CalledProcessError:
        print("Incorrect permissions on your .pem files.")
        print("Try the following:")
        print("-> chmod -R a+rwX <your .pem directory> #<- this will give everyone access to the .pem!")
        print("The commands below will fix the global access issue.")
        print("-> chmod 0600 <directory>/usercert.pem")
        print("-> chmod 0400 <directory>/userkey.pem")

    test = ["ami_atlas", "show", "dataset", "info", "data13_2p76TeV.00219364.physics_MinBias.merge.NTUP_HI.f519_m1313"]
    try: _getcmd(" ".join(test))
    except: pass

