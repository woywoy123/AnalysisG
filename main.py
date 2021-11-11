from Closure.IO import TestIO, TestDir
from Closure.Event import TestEvent, TestParticleAssignment
from Closure.Plotting import TestTops, TestResonance, TestResonanceMassForEnergies, TestRCJetAssignments
from Closure.GNN import TestSimple4TopGNN, TestDataImport, TestGraphObjects, TestComplex4TopGNN, Test4TopGNNInvMass, TestRCJetAssignmentGNN, Helper
from Closure.DataLoader import TestSingleFile

if __name__ == '__main__':
    #TestDir()
    #TestIO()
    #TestEvent()
    #TestParticleAssignment()
    #TestTops()
    #TestResonance()
    #TestGraphObjects()
    #TestDataImport()
    #TestDataImport("TruthChildren")
    #TestSimple4TopGNN()
    #Test4TopGNNInvMass()
    #TestComplex4TopGNN()
    #TestResonanceMassForEnergies()
    #TestRCJetAssignments()
    #TestRCJetAssignmentGNN()
    #Helper()
    TestSingleFile()
    
