from .Settings import Settings

def _getcmd(cmd):
    import subprocess
    return subprocess.check_output(cmd, stderr = subprocess.STDOUT, shell = True).decode("UTF-8")

def CONFIG_PYAMI(): 
    print("------- CONFIGURING PYAMI -------")
    print(_getcmd('echo "y" | ami_atlas_post_install'))
    print("-------- FINISHED PYAMI ---------")   

def POST_INSTALL_PYTORCH( device = "" ):
    print(_getcmd("pip install torch --index-url https://download.pytorch.org/whl/" + device))

def POST_INSTALL_PYG( device = "" ):
    _pkgs = ["pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv"]
    _torch = "pip install " + " ".join(_pkgs) + " -f https://data.pyg.org/whl/torch-2.0.0+" + device + ".html"
    print(_getcmd(_torch))
    print(_getcmd("pip install torch_geometric"))

def POST_INSTALL_PYC():
    def stdout():
        for line in p.stdout: print(line.decode("UTF-8").replace("\n", ""))
    import subprocess
    from threading import Thread
    print("------- CONFIGURING TORCH-EXTENSIONS (PYC) -------")
    p = subprocess.Popen(["pip", "install", "-v", "./torch-extensions/"], stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    t = Thread(target = stdout, daemon = True)
    t.start()
    t.join()
    print("------- DONE CONFIGURING TORCH-EXTENSIONS (PYC) -------")

def CHECK_CUDA():
    gcc_cu = { 11  : ["11.7", "11.8"]}
    nvcc = _getcmd("nvcc --version | grep release | awk '{print $5}'")[:-2]
    this_gcc = [i for i in gcc_cu if nvcc in gcc_cu[i]]
    gcc = _getcmd("gcc --version | grep gcc | awk '{print $3}'").split("-")[0]
    if len(this_gcc) > 0 and float(this_gcc[0]) <= this_gcc[0]: cu = "cu" + nvcc.replace(".", "")
    else: cu = "cpu" 
    print("------- CONFIGURING PYTORCH -------")
    print("+++++> Suggested Installer: " + cu + " (gcc: " + gcc + ")")
    POST_INSTALL_PYTORCH(cu)
    print("-------- FINISHED PYTORCH ---------")   

    print("------- CONFIGURING PYG -------")
    print("+++++> Suggested Installer: " + cu + " (gcc: " + gcc + ")")
    POST_INSTALL_PYG(cu)
    print("-------- FINISHED PYG ---------")   


