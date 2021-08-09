from Closure.IO import TestIO, TestDir
from Closure.Event import TestEvent, TestParticleAssignment
from Closure.Plotting import TestTops, TestResonance
from Closure.GNN import TestSimple4TopGNN, TestDataImport, TestGraphObjects

if __name__ == '__main__':
    TestDir()
    TestIO()
    TestEvent()
    TestParticleAssignment()
    TestTops()
    TestResonance()
    TestGraphObjects()
    TestDataImport()
    TestDataImport("TruthChildren")
    #TestSimple4TopGNN()
