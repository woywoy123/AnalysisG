import os
from glob import glob
from Functions.Tools.Alerting import *

class Directories(Notification):
    def __init__(self, dir, Verbose = True):
        Notification.__init__(self, Verbose)
        self.Caller = "Directories"
        
        self.__Dir = dir 
        self.Notify("READING +-> " + dir)
        self.__pwd = os.getcwd()
        self.Files = {}

    def ListDirs(self):
        os.chdir(self.__Dir)
        for i in glob("*"):
            if os.path.isdir(i):
                self.Files[os.getcwd()+"/"+i] = []
        os.chdir(self.__pwd)

    def ListFilesInDir(self, dir):
        os.chdir(dir)
        Output = []
        for i in glob("*"):
            if os.path.isfile(i) and ".root" in i:
                Output.append(i)
        os.chdir(self.__pwd)
        return Output

    def GetFilesInDir(self):
        self.ListDirs()
        for dir in self.Files:
            self.Files[dir] = self.ListFilesInDir(dir)

