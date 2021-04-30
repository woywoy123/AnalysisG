from BaseFunctions.IO import *

def ReadLeafsFromResonance(file_dir):

    Entry_Objects = ObjectsFromFile(file_dir, "nominal", ["top_FromRes", "truth_top_pt"])
   
    # Continue here and read the branch and see what the contents are.
    print(Entry_Objects)
