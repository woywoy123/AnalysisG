from Closure.IO import TestDir, TestReadSingleFile, TestReadFile, TestFileConvertArray
from Closure.Event import TestEvents



#from Closure.Event import TestEvent, TestParticleAssignment
#from Closure.Plotting import TestTops, TestResonance, TestResonanceMassForEnergies, TestRCJetAssignments, TestBackGroundProcesses
#from Closure.GNN import TestSimple4TopGNN, TestDataImport, TestGraphObjects, TestComplex4TopGNN, Test4TopGNNInvMass, TestRCJetAssignmentGNN
#from Closure.DataLoader import TestSignalSingleFile, TestSignalMultipleFile, TestSignalDirectory, TestSingleTopFile

def Passed(F, name):
    if F:
        print("(+)Passed: "+name)
    else:
        print("(-)Failed: "+name)


if __name__ == '__main__':
    # ====== Test of IO 
    #Passed(TestDir(), "TestDir")
    #Passed(TestReadSingleFile(), "TestReadSingleFile")
    #Passed(TestReadFile(), "TestReadFile")
    #Passed(TestFileConvertArray(), "TestFileConvertArray")
   
    # ====== Test of EventGenerator 
    Passed(TestEvents(), "TestEvent")

    #TestEvent()
    #TestParticleAssignment()
    #TestTops()
    #TestResonance()
    #TestSignalSingleFile()
    #TestSignalMultipleFile()  
    #TestSignalDirectory()
    #TestSingleTopFile()
    #TestBackGroundProcesses()


    # ====== DataLoader Tests
    #TestGraphObjects()
    #TestDataImport()
    #TestDataImport("TruthChildren")
    #TestSimple4TopGNN()
    #Test4TopGNNInvMass()
    #TestComplex4TopGNN()
    #TestResonanceMassForEnergies()
    #TestRCJetAssignments()
    #TestRCJetAssignmentGNN()

