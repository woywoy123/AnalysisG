from glob import glob 
import os 
import uproot
import threading
import numpy as np
import pickle

class FastReading():
    def __init__(self, File_dir):
        self.File_dir = File_dir
        self.OutputTree = []
        self.FilesTree = []
        self.OutputBranchFromTree = []
        self.FilesBranchFromTree = []
        self.ActiveThreads = []
        self.ArrayBranches = {}
        self.FindFilesInDir()
        self.Verbose = True
        self.a = 0
        self.i = 0
        self.Interval = 10
        
    def GetFilesInDir(self, direct):
        print("INFO::Reading Files and Directories from: " + direct) 
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
        if len(self.GetFilesInDir(direct)) > 0:
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
                    break
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
        print("INFO::Reading given branches from trees")
        
        for i in self.FilesDict:
            for f in self.FilesDict[i]:
                x = UPROOT_Reader(f)
                self.OutputBranchFromTree.append(x.ReadBranchesOrTree(Tree, Branch))
                self.FilesBranchFromTree.append(str(f)) 

    def ConvertBranchesToArray(self, Branch = [], core = 4):
        print("INFO::Converting branch objects to arrays")
        
        def StartThreads( File_Index, Branch):
            for tr in self.OutputBranchFromTree[File_Index]:
                for k in range(len(self.OutputBranchFromTree[i][tr])):
                    br = self.OutputBranchFromTree[i][tr][k]
                    if isinstance(br, str):
                        continue
                    if str(br.name) in Branch:
                        x = threading.Thread(target = self.ConvertToArray, args=(File_Index, tr, k))
                        self.ActiveThreads.append(x)

                    elif len(Branch) == 0:
                        x = threading.Thread(target = self.ConvertToArray, args=(File_Index, tr, k))
                        self.ActiveThreads.append(x)
            
        if len(self.OutputBranchFromTree) == 0:
            return "No Branches Selected!"
        
        # Get from list of each file 
        for i in range(len(self.OutputBranchFromTree)): 
            if isinstance(Branch, list) and len(self.OutputBranchFromTree) != 0:
                StartThreads(i, Branch)    
            elif isinstance(Branch, str):
                StartThreads(i, [Branch])
        
        for th in self.ActiveThreads:
            if len(threading.enumerate()) >= core:
                while True:
                    if len(threading.enumerate()) < core:
                        break
                    
            th.start()

        while True:
            if len(threading.enumerate()) == 1:
                break

        print("INFO::Finished Converting Branches to Arrays")

    def ConvertToArray(self, File_Index, Tree, Branch):
        # Check that the Branch doesnt contain "NotFound" entries
        
        name = "NaFailed"
        try:
            name = self.OutputBranchFromTree[File_Index][Tree][Branch].name
        except: 
            pass    

        conv = "BrFailed"
        try:
            conv = np.array(self.OutputBranchFromTree[File_Index][Tree][Branch].array(library = "np"))
        except:
            pass

        try: 
            self.ArrayBranches[Tree]
        except KeyError:
            self.ArrayBranches[Tree] = {}

        try:
            self.ArrayBranches[Tree][name] = np.concatenate((self.ArrayBranches[Tree][name], conv))
        except KeyError:
            self.ArrayBranches[Tree][name] = conv
        
        diction = {}
        diction[name] = conv
        self.OutputBranchFromTree[File_Index][Tree][Branch] = diction

        self.i = self.i + 1
        self.ProgressAlert()

    def ProgressAlert(self):

        if self.Verbose == True:
            
            per = round(float(self.i) / float(len(self.ActiveThreads))*100)
            if  per > self.a:
                print("INFO::Progress " + str(per) + "%")
                self.a = self.a + self.Interval











class UPROOT_Reader():
    def __init__(self, File):
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
            return "NotFound"


def PickleObject(obj, filename):
    
    outfile = open("./PickledObjects/"+filename, "wb")
    pickle.dump(obj, outfile)
    outfile.close()

def UnpickleObject(filename):

    infile = open("./PickledObjects/"+filename, "rb")
    obj = pickle.load(infile)
    infile.close()
    return obj
