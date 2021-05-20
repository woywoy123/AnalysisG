from BaseFunctions.IO import ListSampleDirectories, ObjectsFromFile, SpeedOptimization

def TestDirectoryList(entry_dir):

    # Do Closure Tests here for the IO
    Files_dir = ListSampleDirectories(entry_dir)
    for key in Files_dir:
        for f in Files_dir[key]:
            print(f)

def TestObjectsFromFile(entry_dir):
    
    def Assessment(out, TestCase):
        for i in out:

            input_folder = i
            root_files_map = out[i]
            
            assert(isinstance(input_folder, str))
            assert(isinstance(root_files_map, dict))
            
            print( " --> " + input_folder)
            for x in root_files_map:
                tree_in_root = x 
                trees = root_files_map[x]
                
                assert(isinstance(tree_in_root, str))
                assert(isinstance(trees, dict))
                
                print( " --> " + tree_in_root)
                for p in trees:
                    tree_name = p
                    branch_list = trees[p]

                    assert(isinstance(tree_name, str))
                    assert(isinstance(branch_list, list))

                    branch_dict = branch_list[1]

                    assert(isinstance(branch_dict, dict))
                    
                    print( " --> " + tree_name)
                    for b in branch_dict:
                        branch_name = b
                        branch_obj = branch_dict[b]

                        assert(isinstance(branch_name, str))
                        print( " --> " + branch_name)
            
            print("Passed case: " + TestCase)


    # Get example
    Files = ListSampleDirectories(entry_dir)
    

    #Test the case where I am simply reading a dict with a root directory + sub root files
    out = ObjectsFromFile(Files, "nominal", "mu")
    Assessment(out, "ROOT_Folder+ROOT_Files+SingleTree+SingleBranch")

    out = ObjectsFromFile(Files, ["nominal", "truth"], "mu")
    Assessment(out, "ROOT_Folder+ROOT_Files+MultipleTree+SingleBranch")

    out = ObjectsFromFile(Files, ["nominal", "truth"], ["mu", "el_e"])
    Assessment(out, "ROOT_Folder+ROOT_Files+MultipleTree+MultipleBranch (one of the branches isnt actually there)")


def TestSpeedOptimization():
    files = "/dev/shm/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root"
    SpeedOptimization(files)

    
        


            
        

        
