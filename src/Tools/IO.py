from AnalysisG.Notification import _IO
from .String import *
from .Miscellaneous import *
import os
import copy
from glob import glob
from AnalysisG._Tools import Hash

class Code:

    def __init__(self, Instance):
        self._Name = None 
        self._Module = None 
        self._Path = None 
        self._Code = None 
        self._Hash = None
        self._File = None
        self.DumpCode(Instance)
        self._Instance = Instance

    def DumpCode(self, Instance):
        try: self._Name = Instance.__qualname__
        except AttributeError:
            try: self._Name = Instance.__name__
            except AttributeError: self._Name = type(Instance).__name__
        try: self._Module = Instance.__module__
        except AttributeError: self._Module = Instance.__package__

        self._Path = self._Module + "." + self._Name
        self._FileCode = GetSourceFile(Instance)
        self._Code = GetSourceCode(Instance)
        self._Hash = Hash(self._Code)
        self._File = GetSourceFileDirectory(Instance)
    
    @property
    def clone(self):
        Instance = self._Instance
        if callable(Instance):
            try: Inst = Instance()
            except: Inst = Instance
            Inst = Instance
        _, inst = StringToObject(self._Module, self._Name)
        return inst

    def __eq__(self, other):
        return self._Hash == other._Hash

class IO(String, _IO):

    def __init__(self):
        self.Caller = "IO"
        self.Verbose = 3
    
    def lsFiles(self, directory, extension = None):
        srch = glob(directory + "/*") if extension == None else glob(directory + "/*" + extension)
        srch = [i for i in srch]
        if len(srch) == 0: self.EmptyDirectoryWarning(directory)
        return srch
    
    def ls(self, directory):
        try: return os.listdir(directory)
        except OSError: return []

    def IsFile(self, directory):
        if os.path.isfile(directory): return True
        self.FileNotFoundWarning(self.path(directory), directory.split("/")[-1])
        return False

    def ListFilesInDir(self, directory, extension, _it = 0):
        F = []
        directory = copy.deepcopy(directory)
        if isinstance(directory, dict):
            for i in directory:
                if i.endswith(extension): F += [i]
                elif isinstance(directory[i], list): directory[i] = [i + "/" + k for k in directory[i]]
                else: directory[i] = [i + "/" + directory[i]]
                F += self.ListFilesInDir([k for k in self.ListFilesInDir(directory[i], extension, _it+1)], extension, _it+1)
        elif isinstance(directory, list):
            F += [t for k in directory for t in self.ListFilesInDir(k, extension, _it+1)]
        elif isinstance(directory, str):
            if directory.endswith("*"): F += self.lsFiles(directory[:-2], extension)
            elif directory.endswith(extension): F += [directory]
            elif len(self.lsFiles(directory, extension)) != 0: F += self.lsFiles(directory, extension)
            F = [i for i in F if self.IsFile(i)] 

        if _it == 0:
            dirs = {self.path(i) : [] for i in F}
            F = {i : [k.split("/")[-1] for k in F if self.path(k) == i] for i in dirs} 
            Out = {}
            for i in F:
                if len(F[i]) == 0:
                    self.EmptyDirectoryWarning(i)
                    continue
                Out[i] = F[i]
            self.FoundFiles(Out)
            return Out
        return F
    
    @property
    def pwd(self): return os.getcwd()

    def abs(self, directory): return os.path.abspath(directory)
    
    def path(self, inpt): return os.path.dirname(self.abs(inpt))
    
    def filename(self, inpt): return inpt.split("/")[-1]

    def mkdir(self, directory):
        try: os.makedirs(self.abs(directory))
        except FileExistsError: pass

    def rm(self, directory):
        try: os.remove(self.abs(directory))
        except IsADirectoryError:
            import shutil
            shutil.rmtree(self.abs(directory))
        except FileNotFoundError: pass

    def cd(self, directory):
        os.chdir(directory)
        return self.pwd()
    
    def Hash(self, string): return Hash(string)




