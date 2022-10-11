import importlib
import inspect

def StringToObject(module, name):
    modul = importlib.import_module(module)
    obj = getattr(modul, name)
    return CheckObjectInputs(obj)



def CheckObjectInputs(obj):
    InptDic = {}
    _def = obj.__init__.__defaults__
    if _def == None:
        return InptDic, obj()
    _inptvars = list(obj.__init__.__code__.co_varnames)
    _inptvars.remove("self")
    
    print("Fix" , _inptvars, _def)



def GetSourceCode(obj):
    return inspect.getsource(obj)




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
