import importlib


def StringToObject(string):
    
    print(string)

def RecallObjectFromString(string):
    import importlib
    mod = importlib.import_module(".".join(string.split(".")[:-1]))
    Reco = getattr(mod, string.split(".")[-1])
    
    if Reco.__init__.__defaults__ == None:
        return Reco()

    def_inp = list(Reco.__init__.__defaults__)
    inp = list(Reco.__init__.__code__.co_varnames)
    inp.remove("self")
    
    l = {i : None for i in inp}
    
    inp.reverse()
    def_inp.reverse()

    for i, j in zip(inp, def_inp):
        l[i] = j
    
    return Reco(**l)
