from AnalysisTopGNN.Tools import IO


def TestDirectory(Files):
    D = {Files + "/tttt" : ["*"], Files + "/t" : "QU_14.root", Files + "/ttbar/" : "*"}
    F = IO(D, ".root")
    
    return True

