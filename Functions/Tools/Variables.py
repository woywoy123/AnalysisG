
class VariableManager:

    def __init__(self):
        pass

    def ListAttributes(self):
        self.Leaves = []
        for i in list(self.__dict__.values()):
            if isinstance(i, str) and i != "" and i != self.Type:
                self.Leaves.append(i)

    def SetAttribute(self, key, val):
        setattr(self, key, val)

    def CompileKeyMap(self):
        self.KeyMap = {}
        for i in self.__dict__.keys():
            val = self.__dict__[i]
            if isinstance(val, str) and self.Type != val:
                self.KeyMap[val] = i

    def GetAttributeFromKeyMap(self, key):
        obj = getattr(self, self.KeyMap[key])
        return obj
    
    def ClearVariable(self):
        del self.Leaves 
        del self.KeyMap
        
        


