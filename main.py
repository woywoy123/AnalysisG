from Closure.IO import TestDir, TestReadSingleFile, TestReadFile, TestFileConvertArray
from Closure.Event import TestEvents, TestParticleAssignment, TestSignalMultipleFile, TestSignalDirectory
from Closure.Plotting import TestTops, TestResonance


#from Closure.Plotting import TestTops, TestResonance, TestResonanceMassForEnergies, TestRCJetAssignments, TestBackGroundProcesses
#from Closure.GNN import TestSimple4TopGNN, TestDataImport, TestGraphObjects, TestComplex4TopGNN, Test4TopGNNInvMass, TestRCJetAssignmentGNN
#from Closure.DataLoader import TestSignalSingleFile, TestSignalMultipleFile, TestSignalDirectory, TestSingleTopFile


def Passed(F, name):
    if F:
        print("(+)Passed: "+name)
    else:
        print("(-)Failed: "+name)


dir = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root"
def Generate_Cache():
    from Functions.Event.Event import EventGenerator
    from Functions.IO.IO import PickleObject
    ev = EventGenerator(dir)
    ev.SpawnEvents()
    ev.CompileEvent(SingleThread = False)
    compiler = "EventGenerator"
    PickleObject(ev, compiler)


if __name__ == '__main__':
    # ====== Test of IO 
    #Passed(TestDir(), "TestDir")
    #Passed(TestReadSingleFile(), "TestReadSingleFile")
    #Passed(TestReadFile(), "TestReadFile")
    #Passed(TestFileConvertArray(), "TestFileConvertArray")
    #Passed(TestSignalMultipleFile(), "TestSignalMultipleFile")
    Passed(TestSignalDirectory(), "TestSignalDirectory")

    # ====== Test of EventGenerator 
    #Generate_Cache()
    #Passed(TestEvents(), "TestEvents")
    #Passed(TestParticleAssignment(), "TestParticleAssignment")
    #Passed(TestTops(), "TestTops")
    #Passed(TestResonance(), "TestResonance")


    #TestSignalSingleFile()
    #TestSignalMultipleFile()  
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

