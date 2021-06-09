from glob import glob 
import os 
import uproot
import multiprocessing
import numpy as np
import pickle
from time import sleep
from BaseFunctions.Alerting import ErrorAlert

class FastReading():
    def __init__(self, File_dir):
        self.File_dir = File_dir
        self.OutputTree = []
        self.FilesTree = []
        self.OutputBranchFromTree = []
        self.FilesBranchFromTree = []
        self.FindFilesInDir()
        self.Verbose = True
        self.a = 0
        self.i = 0
        
        self.ActiveThreads = []
        self.ConvertedArray = {}
        self.ArrayBranches = {}

        self.Interval = 10
        
    def GetFilesInDir(self, direct, verb = False):
        if verb:
            print("FASTREADING::INFO::Reading Files and Directories from: " + direct) 
        files = [] 
        for i in glob(direct + "/*"):
            files.append(i)
        
        return files
    
    def GetFilesInSubDir(self, direct):
        files = []
        for i in self.GetFilesInDir(direct):
            if ".root" in i:
                files.append(i)
        return files

    def FindFilesInDir(self):
        Output = {}
        direct = "" 
        if self.File_dir[len(self.File_dir) -1:len(self.File_dir)] == "/":
            direct = self.File_dir[0:len(self.File_dir)-1]
        else:
            direct = self.File_dir
       
        # Check if the given dir is really a directory:
        isDir = False
        if len(self.GetFilesInDir(direct, True)) > 0:
            isDir = True
        else:
            isDir = False
        
        # Check if given dir has subdirectories
        if isDir:
            for i in self.GetFilesInDir(direct):
                inSub = self.GetFilesInSubDir(i)
                if len(inSub) > 0: 
                    Output[i] = []
                elif ".root" in i:
                    Output[direct] = self.GetFilesInSubDir(direct)
                    print("FASTREADING::INFO::Found Files: " + i)
                for x in inSub:
                    Output[i].append(x)
        
        else:
            Output["File"] = [direct]
        
        self.FilesDict = Output
        return Output   
    
    def ReadTree(self, Tree):
        for i in self.FilesDict:
            for f in self.FilesDict[i]:
                x = UPROOT_Reader(f)
                self.OutputTree.append(x.ReadBranchesOrTree(Tree = Tree))
                self.FilesTree.append(str(f))

    def ReadBranchFromTree(self, Tree, Branch):
        print("FASTREADING::INFO::Reading given branches from trees")
        
        for i in self.FilesDict:
            for f in self.FilesDict[i]:
                x = UPROOT_Reader(f)
                self.OutputBranchFromTree.append(x.ReadBranchesOrTree(Tree, Branch))
                self.FilesBranchFromTree.append(str(f)) 

    def ConvertBranchesToArray(self, Branch = []):
        print("FASTREADING::INFO::Converting branch objects to arrays")
        
        def Convert(sender, obj):
            ret = obj.array(library = "np")
            sender.send(ret) 

        def InterfaceHandler(B, objb):
            # Skip the case where there was an error with the reading
            skip = False 
            if isinstance(objb, str):
                skip = True
            
            # Skip all Branches not quoted by given string 
            if isinstance(B, str):
                if str(objb.name != B):
                    skip = True
            
            # Skip all Branches not in list
            if isinstance(B, list):
            
                if len(B) != 0:
                    if str(objb.name) not in B:
                        skip = True

            return skip 
        
        def SafeDict(dic, key, cntr = {}):
            try:
                dic[key]
            except KeyError:
                dic[key] = cntr

        for f_i in range(len(self.OutputBranchFromTree)):
            for tree in self.OutputBranchFromTree[f_i]:
                SafeDict(self.ConvertedArray, tree)
                
                for branch in self.OutputBranchFromTree[f_i][tree]:
                    SafeDict(self.ConvertedArray[tree], str(branch.name), [])
                   
                    if InterfaceHandler(Branch, branch):
                        continue

                    recv, send = multiprocessing.Pipe(False)
                    p = multiprocessing.Process(target = Convert, args = (send, branch, ))
                    self.ActiveThreads.append(p) 
                    
                    self.ConvertedArray[tree][str(branch.name)].append(recv)
       
        for p in self.ActiveThreads:
            p.start()

        for tr in self.ConvertedArray:
            SafeDict(self.ArrayBranches, tr)
            for br in self.ConvertedArray[tr]:
                SafeDict(self.ArrayBranches[tr], br)
                for i in self.ConvertedArray[tr][br]:
                    array = i.recv()
                    if isinstance(self.ArrayBranches[tr][br], list):
                        self.ArrayBranches[tr][br] = array
                    else:
                        x = list(self.ArrayBranches[tr][br])
                        y = list(array)
                        z = x + y
                        self.ArrayBranches[tr][br] = np.array(z)
                    self.i += 1
                    self.ProgressAlert()
         
        for i in self.ActiveThreads:
            i.join()

        print("FASTREADING::INFO::Finished Converting Branches to Arrays")
    def ProgressAlert(self):

        if self.Verbose == True:
            
            if len(self.ActiveThreads) == 0:
                return 
            per = round(float(self.i) / float(len(self.ActiveThreads))*100)
            if  per > self.a:
                print("INFO::Progress " + str(per) + "%")
                self.a = self.a + self.Interval




class UPROOT_Reader(ErrorAlert):
    def __init__(self, File):
        ErrorAlert.__init__(self)

        self.ROOT_Original = ""
        self.ROOT_Original = uproot.open(File)
        self.FoundTrees = {}
        self.FoundBranchesInTrees = {}
        self.CurrentTree = ""
        self.CurrentBranch = ""
        self.ROOT_F_Current = self.ROOT_Original
        self.OutputState = {}

    def SafeCheck(self, key, Uproot):
        Avail = False
        try:
            Uproot[key]
            Avail = True
        except uproot.exceptions.KeyInFileError:
            Avail = False
        return Avail
    
    def ReadBranchesOrTree(self, Tree = -1, Branch = -1):

        # Case 1.1: Only a Tree is given but a list
        if isinstance(Tree, list) and Branch == -1:
            for i in Tree:
                self.CurrentTree = i
                self.OutputState[i] = self.ReadObject()
      
        # Case 1.2: Only a Tree is given but is string 
        elif isinstance(Tree, str) and Branch == -1:
            self.CurrentTree = Tree
            self.OutputState[Tree] = self.ReadObject()
        
        # Case 2.1 : Tree and Branch are both given in list format
        elif isinstance(Tree, list) and isinstance(Branch, list):
            for i in Tree:
                self.OutputState[i] = []
                for j in Branch:
                    self.CurrentTree = i
                    self.CurrentBranch = j
                    self.OutputState[i].append(self.ReadObject())
        
        #Case 2.2 : Tree is string and Branch is List
        elif isinstance(Tree, str) and isinstance(Branch, list):
            self.CurrentTree = Tree
            self.OutputState[Tree] = []
            for i in Branch:
                self.CurrentBranch = i
                self.OutputState[Tree].append(self.ReadObject())

        #Case 2.3 : Tree is List and Branch is string 
        elif isinstance(Tree, list) and isinstance(Branch, str):
            self.CurrentBranch = Branch
            for i in Tree:
                self.CurrentTree = i
                self.OutputState[i].append(self.ReadObject())

        #Case 2.4 : Both Tree and Branch are string 
        elif isinstance(Tree, str) and isinstance(Branch, str):
            self.CurrentBranch = Branch
            self.CurrentTree = Tree
            self.OutputState[Branch] = self.ReadObject()

        return self.OutputState 

    def ReadObject(self):
        passed = False 
        if self.SafeCheck(self.CurrentTree, self.ROOT_Original):
            self.ROOT_F_Current = self.ROOT_Original[self.CurrentTree]
            passed = True
        
        if self.CurrentBranch == "":
            return self.ROOT_F_Current

        if passed and self.SafeCheck(self.CurrentBranch, self.ROOT_F_Current):
            self.ROOT_F_Current = self.ROOT_F_Current[self.CurrentBranch]
            passed = True
        else:
            passed = False
        
        if passed: 
            return self.ROOT_F_Current
        else: 
            self.MissingBranch(self.CurrentBranch, "MISSING BRANCH") 

def PickleObject(obj, filename):
    
    outfile = open("./PickledObjects/"+filename, "wb")
    pickle.dump(obj, outfile)
    outfile.close()

def UnpickleObject(filename):

    infile = open("./PickledObjects/"+filename, "rb")
    obj = pickle.load(infile)
    infile.close()
    return obj
