import os
from glob import glob
from AnalysisTopGNN.Tools import *
import re

class Directories(Notification):
    def __init__(self, directory = None, Verbose = True):
        Notification.__init__(self, Verbose)
        self.Caller = "Directories"
        
        if directory != None:
            self._Dir = directory
            self.Notify("READING +-> " + directory)
        
        self.pwd = os.getcwd()
        self.Files = {}

    def ListDirs(self):
        try:
            os.chdir(self._Dir)
        except NotADirectoryError:
            return None
        
        for i in glob("*"):
            if os.path.isdir(i):
                self.Files[os.getcwd()+"/"+i] = []
            elif os.path.isdir(self._Dir) and os.path.isfile(i):
                self.Files[os.getcwd()] = []
        os.chdir(self.pwd)

    def ListFilesInDir(self, dir):
        accepted = [".root", ".pt", ".pkl", ".hdf5", ".onnx"]
        if dir.endswith("/"):
            dir = dir[:-1]
        Output = {}
        if os.path.isfile(dir) and len([k for k in accepted if dir.endswith(k)]) != 0:
           Output[dir] = ""
        elif len([k for k in accepted if dir.endswith(k)]) != 0:
            self.Warning("FILE: " + dir + " NOT FOUND!")

        for i in glob(dir + "/**", recursive = True):
            if os.path.isfile(i) and len([k for k in accepted if i.endswith(k)]) != 0:
                Output[i] = ""
            else:
                continue
        integers = { re.search(r'\d+$', ".".join(i.split(".")[:-1])) : i  for i in Output }
        integers = { int(i.group()) : integers[i] for i in integers if i != None }
        integers = { integers[i] : "" for i in sorted(integers) }
        Output = {i : "" for i in Output if i not in integers} 
        Output |= integers 

        for i in Output:
            self.Notify("!!FOUND +-> " + i.replace("//", "/"))
        return list(Output)

    def GetFilesInDir(self):
        self.Files = {}
        for i in self.ListFilesInDir(self._Dir):
            Filename = i.split("/")[-1]
            FileDir = "/".join(i.split("/")[:-1])
            if FileDir not in self.Files:
                self.Files[FileDir] = []
            self.Files[FileDir].append(Filename)

class WriteDirectory(Notification):
    def __init__(self):
        self.pwd = os.getcwd()
        self.__tmp = ""

    def MakeDir(self, Dir):
        try:
            os.makedirs(Dir, exist_ok = True)
        except:
            self.Warning("SOMETHING WENT WRONG MAKING DIR! -> " + Dir)

    def ChangeDirToRoot(self, Dir = None):
        if Dir != None:
            self.pwd = Dir
        os.chdir(self.pwd)
    
    def ChangeDir(self, dir):
        os.chdir(self.pwd + "/" + dir)
    
    def WriteTextFile(self, inp, di_, name):
        self.MakeDir(self.pwd + "/" + di_)
        if name.endswith(".txt") != True:
            name += ".txt"
        f = open(self.pwd + "/" + di_ + "/" + name, 'w')
        if isinstance(inp, list):
            inp = "\n".join(inp)
        f.write(inp)
        f.close()



