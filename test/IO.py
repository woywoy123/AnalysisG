from AnalysisTopGNN.Tools import Tools

def TestlsFiles(Directory):
    I = Tools()
    
    print("Testing listing sub-directories...")
    F = I.lsFiles(Directory)
    for i in range(3):
        assert F[2-i] == Directory + "Dir" + str(i+1)
    print("Testing listing files with extension .txt")
    assert len(I.lsFiles(Directory + "Dir1", ".txt")) == 3
    return True

def Testls(Directory):
    I = Tools()
    print("Testing listing directories")
    F = I.lsFiles(Directory)
    for i in range(3):
        assert F[2-i] == Directory + "Dir" + str(i+1)
    if I.ls("FakeDirectory") != []:
        return False
    return True

def TestIsFile(Directory):
    I = Tools()
    for i in range(3):
        assert I.IsFile(Directory + "Dir" + str(i+1) + "/" + str(i+1) + ".txt") == True
    assert I.IsFile(Directory + "Dir1") == False
    return True

def TestListFilesInDir(Directory):
    D = {Directory + "/Dir1" : ["1.txt"], Directory + "/Dir2" : "2.txt", Directory + "/Dir3/" : "*"}
    I = Tools()
    O = I.ListFilesInDir(D, ".txt")
    assert "1.txt" in O[I.abs(Directory + "/Dir1")]
    assert "2.txt" in O[I.abs(Directory + "/Dir2")]
    assert "1.txt" in O[I.abs(Directory + "/Dir3")]
    assert "2.txt" in O[I.abs(Directory + "/Dir3")]
    assert "3.txt" in O[I.abs(Directory + "/Dir3")]
    return True

def TestSourceCodeExtraction():

    class HelloWorld:
        def __init__(self, hello, world = "world"):
            pass

        def Test(self):
            return True

    T = Tools()
    H = HelloWorld("H")
    print(T.GetSourceCode(H))
    print("")
    print(T.GetSourceFile(H))
    print("")
    return True

def TestDataMerging():
    
    T = Tools()
    d = {"1": [["1", "2"], ["3", "4"], ["5", "6"], [["2"]]], "2": ["5", "6"], "3": [["1", "2"], ["3", "4"]]}
    assert T.MergeListsInDict(d) == ["1", "2", "3", "4", "5", "6", "2", "5", "6", "1", "2", "3", "4"]
    l = [["1", "2"], ["3", "4"], ["5", "6"], [["2"]]]
    assert T.MergeNestedList(l) == ["1", "2", "3", "4", "5", "6", "2"]
    return True
