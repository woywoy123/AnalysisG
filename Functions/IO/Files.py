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
            else:
                self.Files[os.getcwd()] = []
        os.chdir(self.__pwd)

    def ListFilesInDir(self, dir):
        os.chdir(dir)
        Output = []
        for i in glob("*"):
            if os.path.isfile(i) and ".root" in i:
                Output.append(i)
                self.Notify("FOUND +-> " + i)
        os.chdir(self.__pwd)
        return Output

    def GetFilesInDir(self):
        if os.path.isfile(self.__Dir) == False:
            self.ListDirs()
            for dir in self.Files:
                self.Files[dir] = self.ListFilesInDir(dir)
        else:
            self.Files[os.path.dirname(self.__Dir)] = [os.path.basename(self.__Dir)]

class WriteDirectory(Notification):
    def __init__(self):
        self.__pwd = os.getcwd()

    def MakeDir(self, dir):
        try:
            os.mkdir(self.__pwd + "/" + dir)
        except FileExistsError:
            pass

    def ChangeDirToRoot(self):
        os.chdir(self.__pwd)
    
    def ChangeDir(self, dir):
        os.chdir(self.__pwd + "/" + dir)
