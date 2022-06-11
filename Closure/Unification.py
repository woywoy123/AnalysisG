from Functions.Unification.Unification import Unification 

def TestUnification(FileDir, Files):
    U = Unification()
    #U.AddSample("AllSamples", FileDir)
    U.AddSample("OtherSamples", Files)
    U.Launch()
    return True
