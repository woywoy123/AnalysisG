from glob import glob 
import os 
import uproot
import threading
from time import sleep

def ListSampleDirectories(root):
    out = {}
    
    if root[len(root)-1] == "/":
        root = root[0: len(root) -1]

    # Case where the given directory has other subdirectories
    if len(glob(str(root + "/*/"))) != 0: 
        for i in glob(str(root + "/*/")):
            splitted = i.split("/")
            x = len(splitted)
            filename = splitted[x-2]
            
            out[filename] = []
            for t in glob(str(root + "/" + filename + "/*")):
                if ".root" in t:
                    out[filename].append(t)
                    print(t)
    
    # Case the given directory has files 
    if len(glob(str(root + "/*"))) != 0:
        out[root] = [] 
        for i in glob(str(root + "/*")):
            out[root].append(i) 

    return out


def ObjectsFromFile(*args, **kwds):
    output = {} 
    if len(args) == 1 or len(args) > 3:
        output["WRONGINPUT!"] = ""
        return output

    if len(args) == 1:
        Files = args[0]
        Trees = [] 
        Branches = []
    if len(args) == 2:
        Files = args[0]
        Trees = args[1]
        Branches = []
    if len(args) == 3:
        Files = args[0]
        Trees = args[1]
        Branches = args[2]

    if os.path.isdir(Files):
        Files = ListSampleDirectories(Files)
    
    if isinstance(Files, dict):
        for i in Files:
            d = Files[i]

            if isinstance(d, list):
                sub_file = {} 
                for x in d:
                    sub_file[x] = ReturnTreeFromFile(x, Trees, Branches)
                output[i] = sub_file; 
            
            if isinstance(d, str):
                output[d] = ReturnTreeFromFile(d, Trees, Branches)
            

    if isinstance(Files, list):
        for i in Files:
            output[i] = ReturnTreeFromFile(i, Trees, Branches)

    if isinstance(Files, str):
        output[Files] = ReturnTreeFromFile(Files, Trees, Branches)
   
    return output


def ReturnTreeFromFile(file_dir, trees = [], branches = []):
    
    Output_dict = {}

    # Open the file and check the keys within the ROOT file 
    f = uproot.open(file_dir)
    tree_keys = set(f.keys())

    # Check the data type and adjust input accordingly 
    if isinstance(trees, str):
        trees = [trees] 
    if isinstance(branches, str): 
        branches = [branches]
   
    # Find the tree keys requested in the ROOT File. 
    found = []
    original = []
    tree_obj = {}
    for t in trees:
        
        # Add the versioning tag in the ROOT file 
        original.append(t) 
        if ";1" not in t:
            t = t + ";1"

        if t in tree_keys:
            found.append(t) 
            tree_obj[t] = f[t]
    
    # Case 1: the user has only requested the trees to be returned 
    if len(branches) == 0:
        for i in range(len(found)):
            Output_dict[original[i]] = tree_obj[found[i]]
        return Ouput; 

    
    # Case 2: Get the branches associated with the tree 
    for f in range(len(found)):
        b_keys = set(tree_obj[found[f]].keys())
        
        branch_dict = {}
        for rb in branches:
            if rb in b_keys:
                branch = tree_obj[found[f]][rb] 
                branch_dict[rb] = branch
        Output_dict[original[f]] = [tree_obj[found[f]], branch_dict]
    
    return Output_dict


# This class is going to implement multithreading to solve the horrible reading speeds of uproot 
class FastReading():
    def __init__(self, File_dir):
        self.File_dir = File_dir
        self.OutputTree = []
        self.FilesTree = []
        self.OutputBranchFromTree = []
        self.FilesBranchFromTree = []
        self.ActiveThreads = []
        self.FindFilesInDir()

    def GetFilesInDir(self, direct):
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
            Output["File"] = direct
        
        self.FilesDict = Output
        return Output   
    
    def ReadTree(self, Tree):
        for i in self.FilesDict:
            for f in self.FilesDict[i]:
                x = UPROOT_Reader(f)
                self.OutputTree.append(x.ReadBranchesOrTree(Tree = Tree))
                self.FilesTree.append(str(i+"/"+f))

    def ReadBranchFromTree(self, Tree, Branch):
        for i in self.FilesDict:
            for f in self.FilesDict[i]:
                x = UPROOT_Reader(f)
                self.OutputBranchFromTree.append(x.ReadBranchesOrTree(Tree, Branch))
                self.FilesBranchFromTree.append(str(i+"/"+f)) 

    def ConvertBranchesToArray(self, Branch = [], core = 12):
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
            if len(threading.enumerate()) > core:
                while True:
                    if len(threading.enumerate()) < core:
                        break
                    
                    sleep(1)
            th.start()

        while true:
            if len(threading.enumerate()) == 1:
                break

    def ConvertToArray(self, File_Index, Tree, Branch):
        # Check that the Branch doesnt contain "NotFound" entries
        
        while True:
            try:
                self.OutputBranchFromTree[File_Index][Tree][Branch] = self.OutputBranchFromTree[File_Index][Tree][Branch].array()
                break 
            except: 

                print(self.OutputBranchFromTree[File_Index][Tree][Branch])
                print(File_Index, Tree, Branch)
                continue
        return 0




class UPROOT_Reader():
    def __init__(self, File = ""):
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
                self.OutputState[Tree].apppend(self.ReadObject())

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

         

        


























#def SpeedOptimization(files):
#    
#    def SafeCheck(key, uproot_obj):
#        Avail = False
#        try:
#            uproot_obj[key]
#            Avail = True
#        except uproot.exceptions.KeyInFileError:
#            Avail = False
#        return Avail
#    
#    def ToArray(key, Found_In_File):
#        print("Start", key)
#        Found_In_File[key].array()
#        print("Finish", key)
#
#    folder_files_dict = ListSampleDirectories(files)
#    
#    root_dir = ["nominal"]
#    leaves = ["top_FromRes", "truth_top_charge", "truth_top_e", "truth_top_pt", "truth_top_child_e"]
#    
#    # This is going to be a container 
#    Found_In_File = {}
#    RunThre = []
#    # Iterate through all the folders found from the given directory
#    for i in folder_files_dict:
#        root_files = folder_files_dict[i]
#        Found_In_File[i] = {}
#
#        # Go through individual root files in the directory
#        for x in root_files:
#            root_f = uproot.open(x)
#            Found_In_File[i][x] = {}
#            
#            # Iterate through the requested Tree of the file 
#            for r in root_dir:
#
#                # Check if the given Tree key is in the file (returns false if not found)
#                if SafeCheck(r, root_f) and "uproot.models.TTree" in str(type(root_f[r])):
#                    
#                    Found_In_File[i][x][r] = {}
#                    # See if the given leaves can be found in the root file
#                    for l in leaves:
#                        if SafeCheck(l, root_f[r]) and "uproot.models.TBranch" in str(type(root_f[r][l])):  
#                            Found_In_File[i][x][r][l] = root_f[r][l]
#                            th = threading.Thread(target = ToArray, args = (l, Found_In_File[i][x][r]))
#                            th.start() 
#                            RunThre.append(th) 
#                            if len(RunThre) > 4:
#                                for t in RunThre:
#                                    t.join()
#    
#
#    print(Found_In_File)
