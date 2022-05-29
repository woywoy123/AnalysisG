import os
from glob import glob
from Functions.Tools.Alerting import *
import re

class Directories(Notification):
    def __init__(self, directory = None, Verbose = True):
        Notification.__init__(self, Verbose)
        self.Caller = "Directories"
        
        if directory != None:
            self.__Dir = directory
            self.Notify("READING +-> " + directory)
        
        self.__pwd = os.getcwd()
        self.Files = {}

    def ListDirs(self):
        try:
            os.chdir(self.__Dir)
        except NotADirectoryError:
            return None

        for i in glob("*"):
            if os.path.isdir(i):
                self.Files[os.getcwd()+"/"+i] = []
            elif os.path.isdir(self.__Dir) and os.path.isfile(i):
                self.Files[os.getcwd()] = []
        os.chdir(self.__pwd)

    def ListFilesInDir(self, dir):
        os.chdir(dir)
        if dir.endswith("/"):
            dir = dir[:-1]
        
        Output = {}
        accepted = [".root", ".pt", ".pkl", ".hdf5", ".onnx"]
        for i in glob("*"):
            
            if os.path.isfile(i) and len([k for k in accepted if i.endswith(k)]) != 0:
                pass
            else:
                continue
            Output[i] = ""

        os.chdir(self.__pwd)
        integers = { re.search(r'\d+$', ".".join(i.split(".")[:-1])) : i  for i in Output }
        integers = { int(i.group()) : integers[i] for i in integers if i != None }
        integers = { integers[i] : "" for i in sorted(integers) }
        Output = {i : "" for i in Output if i not in integers} 
        Output |= integers 

        for i in Output:
            self.Notify("!!FOUND +-> " + dir + "/" + i)

        return list(Output)

    def GetFilesInDir(self):
        if os.path.isfile(self.__Dir) == False:
            self.ListDirs()
            for dir in list(self.Files):
                tmp = self.ListFilesInDir(dir)
                if len(tmp) == 0:
                    self.Files.pop(dir)
                    continue
                self.Files[dir] = self.ListFilesInDir(dir)
        else:
            self.Files[os.path.dirname(self.__Dir)] = [os.path.basename(self.__Dir)]

class WriteDirectory(Notification):
    def __init__(self):
        self.__pwd = os.getcwd()
        self.__tmp = ""

    def MakeDir(self, dir):
        self.__tmp = str(self.__pwd)
        for k in dir.split("/"):
            try:
                os.mkdir(self.__tmp + "/" + k)
            except FileExistsError:
                pass
            self.__tmp = self.__tmp + "/" + k 
    def ChangeDirToRoot(self):
        os.chdir(self.__pwd)
    
    def ChangeDir(self, dir):
        os.chdir(self.__pwd + "/" + dir)
    
    def WriteTextFile(self, inp, di_, name):
        self.MakeDir(di_)
        if name.endswith(".txt") != True:
            name += ".txt"
        with open(self.__tmp + "/" + name, 'w') as f:
            f.write(inp)

