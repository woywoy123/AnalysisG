import importlib
import inspect

def StringToObject(module, name):
    modul = importlib.import_module(module)
    obj = getattr(modul, name)
    return CheckObjectInputs(obj)

def CheckObjectInputs(obj):
    _def = obj.__init__.__defaults__
    _req = [i for i in obj.__init__.__code__.co_varnames if i != "self"]
    if _def == None and len(_req) == 0:
        return InptDic, obj()
    
    if _def != None:
        print("Fix" , _def)
    InptDic = {key : None for key in _req} 
    return InptDic, obj  


def GetSourceCode(obj):
    try:
        return inspect.getsource(obj)
    except:
        return insect.getsource(obj.__class__)


def GetSourceFile(obj):
    
    if obj.__class__.__name__ == "type":
        return GetSourceCode(obj)
    return "".join(open(inspect.getfile(obj.__class__), "r").readlines())


#def RecallObjectFromString(string):
#    import importlib
#    mod = importlib.import_module(".".join(string.split(".")[:-1]))
#    Reco = getattr(mod, string.split(".")[-1])
#    
#    if Reco.__init__.__defaults__ == None:
#        return Reco()
#
#    def_inp = list(Reco.__init__.__defaults__)
#    inp = list(Reco.__init__.__code__.co_varnames)
#    inp.remove("self")
#    
#    l = {i : None for i in inp}
#    
#    inp.reverse()
#    def_inp.reverse()
#
#    for i, j in zip(inp, def_inp):
#        l[i] = j
#    
#    return Reco(**l)
