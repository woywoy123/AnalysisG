from Functions.Tools.Alerting import Debugging
from Functions.Tools.Variables import VariableManager
from Functions.Particles.ParticleTemplate import *

class EventTemplate(VariableManager):
    def __init__(self):
        VariableManager.__init__(self)
        self.Type = "Event"
        self.Tree = []
        self.Branches = []

    def DefineObjects(self):
        self.ListAttributes()
        self.CompileKeyMap()
        for i in self.Objects:
           self.SetAttribute(i, {})
        self.MinimalTrees = self.Tree
        self.MinimalLeaves = []
        for i in self.Objects:
            self.MinimalLeaves += self.Objects[i].Leaves
        self.MinimalLeaves += self.Leaves
        self.MinimalBranches = self.Branches

    def ParticleProxy(self, File):
        
        def Attributor(variable, value):
            for i in self.Objects:
                obj = self.Objects[i]
                if variable in obj.KeyMap:
                    o = getattr(self, i)
                    o[variable] = value
                    self.SetAttribute(i, o)
                    return True
            return False

        self.BrokenEvent = False
        self.ListAttributes()
        for i in File.ArrayLeaves:
            if self.Tree not in i:
                continue
            try: 
                val = File.ArrayLeaves[i][self.iter]
            except:
                self.BrokenEvent = True
                continue

            var = i.split("/")[-1]
            if Attributor(var, val):
                continue

            if var in self.KeyMap:
                self.SetAttribute(self.KeyMap[var], val)

    def CompileEvent(self, ClearVal = True):
        pass

    def DictToList(self, inp): 
        out = []
        for i in inp:
            out += inp[i]
        return out
    
    def CompileParticles(self, ClearVal = False):
        for i in self.Objects:
            l = getattr(self, i)
            l = CompileParticles(l, self.Objects[i]).Compile(ClearVal)
            self.SetAttribute(i, l)
 
