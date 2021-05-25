from BaseFunctions.IO import FastReading

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

    
    # Do more tests later 
    
    file_dir = "/CERN/Grid/SignalSamples/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root"

    sample_dirs = "/CERN/Grid/SignalSamples"




