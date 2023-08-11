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
    res = stdout[0].decode("UTF-8")
    print(res)
    if "Cannot" in res: print("Check your pem permissions.")

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

def make_analysis():
    from AnalysisG.Tools import Tools
    tool = Tools()
    this_dir = tool.pwd
    tool.mkdir("Analysis")
    tool.mkdir("Analysis/Objects")

    event = [
            "from AnalysisG.Templates import EventTemplate",
            "from Particles import MyParticle", 
            "",
            "class MyEvent(EventTemplate):",
            "",
            "   def __init__(self)",
            "       EventTemplate.__init__(self)",
            "       self.Objects = {'SomeParticle' : MyParticle}",
            "       self.Trees = ['<some tree>']",
            "       self.Branches = ['<some branches (optional)>']",
            "       self.index = '<some index leaf name>'",
            "       self.CommitHash = '<some hash (optional)>'",
            "",
            "   def CompileEvent(self):",
            "",
            "       # add some matching logic (optional)",
            "       #print the particles in this event",
            "       print(self.SomeParticle)",
    ]

    f = open(this_dir + "/Analysis/Objects/Event.py", "w")
    f.write("\n".join(event))
    f.close()

    particle = [
            "from AnalysisG.Templates import ParticleTemplate",
            "",
            "class MyParticle(ParticleTemplate):",
            "",
            "   def __init__(self):",
            "       ParticleTemplate.__init__(self)",
            "       self.Type = 'some particle type'",
            "       self.pt = self.Type  + '_' + <ROOT leaf string>",
            "       self.eta = self.Type + '_' + <ROOT leaf string>",
            "       self.phi = self.Type + '_' + <ROOT leaf string>",
            "       self.e   = self.Type + '_' + <ROOT leaf string>",
    ]

    f = open(this_dir + "/Analysis/Objects/Particles.py", "w")
    f.write("\n".join(particle))
    f.close()


    graph = [
            "from AnalysisG.Templates import GraphTemplate",
            "",
            "class MyGraph(GraphTemplate):",
            "",
            "   def __init__(self, Event = None):",
            "       GraphTemplate.__init__(self)",
            "       self.Event = Event",
            "       self.Particles += self.Event.SomeParticle",
    ]

    f = open(this_dir + "/Analysis/Objects/Graph.py", "w")
    f.write("\n".join(graph))
    f.close()

    selection = [
            "from AnalysisG.Templates import SelectionTemplate",
            "",
            "class Selection(SelectionTemplate):",
            "",
            "   def __init__(self, Event = None):",
            "       SelectionTemplate.__init__(self)",
            "       self.SomeDictionary = {}",
            "       self.SomeList = []",
            "",
            "   def Selection(self, event):",
            "       return True",
            "",
            "   def Strategy(self, event):",
            "       pass"
    ]

    f = open(this_dir + "/Analysis/Objects/Selection.py", "w")
    f.write("\n".join(selection))
    f.close()

    analysis = [
            "from AnalysisG import Analysis",
            "from Objects.Event import MyEvent",
            "from Objects.Graph import MyGraph", 
            "",
            "Ana = Analysis()",
            "Ana.Event = MyEvent", 
            "Ana.EventGraph = MyGraph", 
            "Ana.EventCache = True # Creates hdf5 files", 
            "Ana.DataCache = True # Creates hdf5 files for graphs", 
            "Ana.Launch()"
    ]

    f = open(this_dir + "/Analysis/main.py", "w")
    f.write("\n".join(analysis))
    f.close()
