from .Miscellaneous import *
from .IO import IO

class Tools(IO):

    def __init__(self):
        IO.__init__(self)
    
    def GetSourceCode(self, obj):
        return GetSourceCode(obj)
    
    def GetObjectFromString(self, module, name):
        return StringToObject(module, name)

    def MergeListsInDict(self, inpt):
        if isinstance(inpt, list):
            return inpt

        out = []
        for i in inpt:
            out += self.MergeListsInDict(inpt[i])
        return out
    
    def DictToList(self, inpt, key = None):
        if isinstance(inpt, str) and key != None:
            return key + "/" + inpt
        if isinstance(inpt, list) and key != None:
            return [self.DictToList(i, key) for i in inpt]
        if isinstance(inpt, dict) and key != None:
            return [self.DictToList(inpt[i], i) for i in inpt]
        if key == None: 
            out = []
            for i in inpt:
                out += self.DictToList(inpt[i], i)
            return out 

