from BaseFunctions.IO import FastReading
import uproot
import numpy as np

def TestSpeedOptimization():
    # Example leaves we want to read  
    leaves = ["top_FromRes", "truth_top_charge", "truth_top_e", "truth_top_pt", "truth_top_child_e"]
    root_dir = ["nominal"]  
   
    # ===== Case 1: A single file is given:
    single_file = "/CERN/Grid/SignalSamples/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root"

    # Initialize the object 
    single = FastReading(single_file)
    assert(list(single.FilesDict.keys())[0] == "File")
    assert(single.FilesDict["File"][0] == single_file)
    print("Single File passed")
    print("")

   
    # Read the Trees of the file
    single_tree = FastReading(single_file)
    single_tree.ReadTree(root_dir)
    assert(len(single_tree.OutputTree) == len(root_dir))
    assert("uproot.models.TTree" in str(type(single_tree.OutputTree[0][root_dir[0]])))
    print("Single File passed: Reading Tree")
    print("")

    # Read the branches of a tree:
    single_tree.ReadBranchFromTree(root_dir, leaves)
    for x in range(len(single_tree.OutputBranchFromTree[0][root_dir[0]])):
        i = single_tree.OutputBranchFromTree[0][root_dir[0]][x]
        assert("uproot.models.TBranch" in str(type(i)))
        assert(i.title == leaves[x])
    print("Single File passed: Reading Leaves from Tree")
    print("")

    test = leaves
    single_tree.ConvertBranchesToArray(Branch = test)
    converted = single_tree.ArrayBranches[root_dir[0]]
    for i in converted:
        assert(i in test)
        assert(isinstance(converted[i], np.ndarray))

    print("Single File passed: Converting Multiple Branches into Numpy Arrays")
    print("")

    
    # ====== Case 2: A file directory is given 
    # Initialize the object 
    multi_files = "/CERN/Grid/SignalSamples/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root"
    multi = FastReading(multi_files)
    assert(list(multi.FilesDict.keys())[0] == multi_files)
    print("Multi File passed")
    print("")

   
    # Read the Trees of the file
    multi_tree = FastReading(multi_files)
    multi_tree.ReadTree(root_dir)
    assert(len(multi_tree.OutputTree) == 2*len(root_dir))
    assert("uproot.models.TTree" in str(type(multi_tree.OutputTree[0][root_dir[0]])))
    print("Multi File passed: Reading Tree")
    print("")

    # Read the branches of a tree:
    multi_tree.ReadBranchFromTree(root_dir, leaves)
    for x in range(len(multi_tree.OutputBranchFromTree[0][root_dir[0]])):
        i = multi_tree.OutputBranchFromTree[0][root_dir[0]][x]
        assert("uproot.models.TBranch" in str(type(i)))
        assert(i.title == leaves[x])
    print("Multiple Files passed: Reading Leaves from Tree")
    print("")

    test = leaves
    multi_tree.ConvertBranchesToArray(Branch = test)
    converted = multi_tree.ArrayBranches[root_dir[0]]
    for i in converted:
        assert(i in test)
        assert(isinstance(converted[i], np.ndarray))

    print("Multiple File passed: Converting Multiple Branches into Numpy Arrays")
    print("")

    # ====== Case 3: A file directory is given with subdirectories
    # Initialize the object 
    sample_dirs = "/CERN/Grid/SignalSamples"
    
    # Read the Trees of the file
    multi_tree = FastReading(sample_dirs)
    multi_tree.ReadTree(root_dir)
    assert("uproot.models.TTree" in str(type(multi_tree.OutputTree[0][root_dir[0]])))
    print("Multi File passed: Reading Tree")
    print("")

    # Read the branches of a tree:
    multi_tree.ReadBranchFromTree(root_dir, leaves)
    for x in range(len(multi_tree.OutputBranchFromTree[0][root_dir[0]])):
        i = multi_tree.OutputBranchFromTree[0][root_dir[0]][x]
        assert("uproot.models.TBranch" in str(type(i)))
        assert(i.title == leaves[x])
    print("Multiple Files passed: Reading Leaves from Tree")
    print("")

    test = leaves
    multi_tree.ConvertBranchesToArray(Branch = test)
    converted = multi_tree.ArrayBranches[root_dir[0]]
    for i in converted:
        assert(i in test)
        assert(isinstance(converted[i], np.ndarray))

    print("Multiple File passed: Converting Multiple Branches into Numpy Arrays")
    print("")


