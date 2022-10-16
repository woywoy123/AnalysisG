import os
from glob import glob
import AnalysisTopGNN
from .String import *

def _IO(directory, extension):
    io = IO()
    return io.ListFilesInDir(directory, extension)

class IO(AnalysisTopGNN.Notification.IO, String):

    def __init__(self):
        self.Caller = "IO"
        self.VerboseLevel = 3

    def lsFiles(self, directory, extension = None):
        srch = glob(directory + "/*") if extension == None else glob(directory + "/*" + extension)
        srch = [i for i in srch]
        if len(srch) == 0:
            self.EmptyDirectoryWarning(directory)
        return srch
    
    def ls(self, directory):
        return os.listdir(directory)

    def IsFile(self, directory):
        if os.path.isfile(directory):
            return directory
        else:
            self.FileNotFoundWarning(self.path(directory, i.split("/")[-1]))
            return False

    def ListFilesInDir(self, directory, extension, _it = 0):
        F = []
        if isinstance(directory, dict):
            for i in directory:
                if isinstance(directory[i], list):
                    directory[i] = [i + "/" + k for k in directory[i]]
                else:
                    directory[i] = [i + "/" + directory[i]]
                F += self.ListFilesInDir([k for k in self.ListFilesInDir(directory[i], extension, _it+1)], extension, _it+1)
        elif isinstance(directory, list):
            F += [t for k in directory for t in self.ListFilesInDir(k, extension, _it+1)]
        elif isinstance(directory, str):
            if directory.endswith("*"):
                F += self.lsFiles(directory[:-2], extension)
            else:
                F += [directory.replace("//", "/")]
            F = [i for i in F if self.IsFile(i)] 

        if _it == 0:
            dirs = {self.path(i) : [] for i in F}
            F = {i : [k.split("/")[-1] for k in F if self.path(k) == i] for i in dirs} 
            Out = {}
            for i in F:
                if len(F[i]) == 0:
                    self.EmptyDirectoryWarning(i)
                Out[i] = F[i]
            self.FoundFiles(Out)
            return Out
        return F
    
    def pwd(self):
        return os.getcwd()

    def abs(self, directory):
        return os.path.abspath(directory)
    
    def path(self, inpt):
        return os.path.dirname(self.abs(inpt))
    
    def filename(self, inpt):
        return inpt.split("/")[-1]

    def mkdir(self, directory):
        try:
            os.makedirs(self.abs(directory))
        except FileExistsError:
            pass

    def cd(self, directory):
        os.chdir(directory)
        return self.pwd()
    


