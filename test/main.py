import sys
import EventGenerator
import IO
import EventGraph
import Exporter
import Submission
import Modular 
import Optimizer

def Test(F, **kargs):
    import traceback
    name = F.__name__
    var = F.__code__.co_varnames
    result = ""
    CallerDir = traceback.format_stack()[0].split("Test(")[1].split("." +name)[0]
    keys = list(set(list(var)).intersection(set(list(kargs))))
    driver = {} 
    for i in keys:
        driver[i] = kargs[i]
    try:
        if F(**driver):
            result = "(+) Passed: " + name + "\n"
        else:
            result = "(-) Failed: " + name + "\n"
    except:
        e = str(sys.exc_info()[1])
        traceback.print_tb(sys.exc_info()[2])
        result = "(-) Failed: " + name + "\n" + e
        print(result)
        exit()
    print(result)


if __name__ == '__main__':

    RootDir = "/home/tnom6927/Downloads/CustomAnalysisTopOutputTest/"
    
    # ==== Testing IO ==== #
    #Test(IO.TestDirectory, Files = RootDir)
    #
    ## ==== Test EventGenerator ==== #
    #Test(EventGenerator.TestEventGenerator, Files = {RootDir + "tttt" : ["QU_0.root"], RootDir + "t" : ["QU_14.root"]})

    ## ==== Test EventGraph ==== #
    #Test(EventGraph.TestEventGraph, Files = [RootDir + "tttt/QU_0.root", RootDir + "t/QU_14.root"])

    ## ==== Test Merger ==== #
    #Test(Exporter.TestEventGenerator, Files = [RootDir + "tttt/QU_0.root", RootDir + "t/QU_14.root"])
    #Test(Exporter.TestGraphGenerator, Files = [RootDir + "tttt/QU_0.root", RootDir + "t/QU_14.root"])
    #Test(Exporter.TestEventGeneratorDumper, Files = [RootDir + "tttt/QU_0.root", RootDir + "t/QU_14.root"])
    #Test(Exporter.TestGraphGeneratorDumper, Files = [RootDir + "tttt/QU_0.root", RootDir + "t/QU_14.root"])

    ## ==== Test Modules ==== #
    #Test(Modular.TestEventGenerator, Files = [RootDir + "tttt/QU_0.root", RootDir + "t/QU_14.root"])

    ## ==== Test Optimizer ==== # 
    #Test(Optimizer.TestOptimizer, Files = [RootDir + "tttt/QU_0.root", {RootDir + "t" : ["QU_14.root", "QU_5.root"]}])

    # === Test Submission ==== #
    #Test(Submission.TestAnalysis, GeneralDir = RootDir)
    Test(Submission.TestCondorDumping, GeneralDir = RootDir)
