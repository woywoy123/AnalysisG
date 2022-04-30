from Closure import IO
from Closure import Event
from Functions.Event import CacheGenerators


def Test(F, **kargs):
    from Functions.IO.Files import WriteDirectory
    import sys
    import traceback
    test_dir = WriteDirectory()
    name = F.__name__
    var = F.__code__.co_varnames
    result = ""
    
    keys = list(set(list(var)).intersection(set(list(kargs))))
    driver = {} 
    for i in keys:
        driver[i] = kargs[i]
    try:
        if F(**driver):
            result = "(+) Passed: " + name
        else:
            result = "(-) Failed: " + name
    except:
        e = str(sys.exc_info()[1])
        traceback.print_tb(sys.exc_info()[2])
        result = "(-) Failed: " + name + "\n" + e
    
    print(result)
    test_dir.WriteTextFile(result, "_TestResults", name)


if __name__ == "__main__":
    GeneralDir = "/CERN/CustomAnalysisTopOutput/"
    
    ## ===== Test IO ===== ##
    #Test(IO.TestDir, di = GeneralDir)
    #Test(IO.TestReadSingleFile, dir_f = GeneralDir + "t/MCa.root") 
    #Test(IO.TestReadFile, di = GeneralDir)
    #Test(IO.TestFileConvertArray, di = GeneralDir)

    ## ===== Test Cache ==== ##
    #Test(CacheGenerators.BuildCacheDirectory, Name = "tttt")
    #CacheGenerators.Generate_Cache(GeneralDir + "tttt/OldSample/1500_GeV/MCe/QU_0.root", Compiler = "tttt")

    ## ===== Test of EventGenerator ===== ##
    #Test(Event.TestEvents, di = GeneralDir + "tttt/OldSample/1500_GeV/MCe/QU_0.root")
    #Test(Event.TestParticleAssignment, di = GeneralDir + "tttt/OldSample/1500_GeV/MCe/QU_0.root")
    #Test(Event.TestSignalMultipleFile, di......) <--- Continue here 
