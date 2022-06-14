from Functions.Unification.Unification import Unification 

def TestUnification(FileDir, Files):
    U = Unification()
    U.Cache = True
    
    for key, Dir in Files.items():
        U.AddSample(key, Dir)
    U.Launch()
    return True
