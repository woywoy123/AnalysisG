from AnalysisTopGNN.Tools import IO


def TestDirectory(Files):
    D = {Files + "/tttt" : ["QU_0.root"], Files + "/t" : "QU_14.root", Files + "/ttbar/" : "*"}
    F = IO(D, ".root")
    print(F) 
    return True

