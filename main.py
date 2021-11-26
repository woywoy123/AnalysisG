from Closure.IO import TestIO, TestDir
from Closure.Event import TestEvent, TestParticleAssignment
from Closure.Plotting import TestTops, TestResonance, TestResonanceMassForEnergies, TestRCJetAssignments, TestBackGroundProcesses
from Closure.GNN import TestSimple4TopGNN, TestDataImport, TestGraphObjects, TestComplex4TopGNN, Test4TopGNNInvMass, TestRCJetAssignmentGNN
from Closure.DataLoader import TestSignalSingleFile, TestSignalMultipleFile, TestSignalDirectory, TestSingleTopFile

if __name__ == '__main__':
    # ====== Test of IO, EventGenerator and Plotting 
    #TestDir()
    #TestIO()
    #TestEvent()
    #TestParticleAssignment()
    #TestTops()
    #TestResonance()
    TestSignalSingleFile()
    TestSignalMultipleFile()  
    TestSignalDirectory()
    TestSingleTopFile()
    TestBackGroundProcesses()


    # ====== DataLoader Tests
    TestGraphObjects()
    TestDataImport()
    TestDataImport("TruthChildren")
    TestSimple4TopGNN()
    Test4TopGNNInvMass()
    TestComplex4TopGNN()
    TestResonanceMassForEnergies()
    TestRCJetAssignments()
    TestRCJetAssignmentGNN()

