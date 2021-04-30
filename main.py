from glob import glob
from ClosureTests.IO import TestDirectoryList, TestObjectsFromFile
from ClosureTests.Plotting import TestSimplePlotting 
from ClosureTests.FourTopsResonance import PlottingResonance


if __name__ == "__main__":
    entry_dir = "/CERN/Grid/Samples"
     
    #TestObjectsFromFile(entry_dir)
    #TestSimplePlotting()
    PlottingResonance()
    

