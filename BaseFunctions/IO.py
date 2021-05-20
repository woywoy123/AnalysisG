from glob import glob 
import os 
import uproot

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

def SpeedOptimization(files):
    
    def SafeCheck(key, uproot_obj):
        Avail = False
        try:
            root_f[key]
            Avail = True
        except uproot.exceptions.KeyInFileError:
            Avail = False
        return Avail




    folder_files_dict = ListSampleDirectories(files)
    
    root_dir = ["nominal"]
    leaves = ["top_FromRes", "truth_top_charge", "truth_top_e", "truth_top_pt", "truth_top_child_e"]
     
    Found_In_File = {}

    for i in folder_files_dict:
        root_files = folder_files_dict[i]
        for x in root_files:
            root_f = uproot.open(x)
                
            for r in root_dir:
                if SafeCheck("nominal", root_f):
                    print(type(root_f[r]))
