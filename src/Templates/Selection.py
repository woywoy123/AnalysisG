from AnalysisG.Templates import ParticleTemplate
from AnalysisG.Tools import Code, Tools
from time import time 
import statistics

try: from PyC.NuSol.Tensors import NuDoublePtEtaPhiE, NuNuDoublePtEtaPhiE
except: pass
try: from PyC.Transform.Floats import Px, Py
except: pass

class Neutrino(ParticleTemplate):
    
    def __init__(self):
        self.Type = "nu"
        ParticleTemplate.__init__(self)

class SelectionTemplate(Tools): 
    def __init__(self):
        self.hash = None
        self.ROOTName = None
        self.Residual = []
        self.CutFlow = {}
        self.TimeStats = []
        self.AllWeights = []
        self.SelWeights = [] 
    
    @property
    def _t1(self): self.__t1 = time()
    
    @property 
    def _t2(self): self.TimeStats.append(time() - self.__t1)
    
    @property
    def AverageTime(self): return statistics.mean(self.TimeStats)
    
    @property
    def StdevTime(self): return statistics.stdev(self.TimeStats)
    
    @property
    def Luminosity(self): return ((sum(self.SelWeights))) / sum(self.AllWeights)
    
    @property
    def NEvents(self): return len(self.AllWeights)

    def Selection(self, event): return True

    def Strategy(self, event): pass

    def Px(self, val, phi): return Px(val, phi)

    def Py(self, val, phi): return Py(val, phi)

    def MakeNu(self, s_):
        if sum(s_) == 0.: return None
        nu = Neutrino()
        nu.px = s_[0]*1000
        nu.py = s_[1]*1000
        nu.pz = s_[2]*1000
        return nu

    def NuNu(self, q1, q2, l1, l2, ev, mT = 172.5, mW = 80.379, mN = 0, zero = 1e-12):
        sol = NuNuDoublePtEtaPhiE(
                q1.pt/1000., q1.eta, q1.phi, q1.e/1000., 
                q2.pt/1000., q2.eta, q2.phi, q2.e/1000., 
                l1.pt/1000., l1.eta, l1.phi, l1.e/1000., 
                l2.pt/1000., l2.eta, l2.phi, l2.e/1000., 
                ev.met/1000., ev.met_phi, 
                mT, mW, mN, zero)
       
        skip = sol[0].tolist()[0]
        _s1 = sol[1].tolist()[0]
        _s2 = sol[2].tolist()[0]
        if skip: return []
        o = [ [self.MakeNu(k), self.MakeNu(j)] for k, j in zip(_s1, _s2) ]
        return [p for p in o if p[0] != None and p[1] != None]
   
    def Nu(self, q1, l1, ev, S = [100, 0, 0, 100], mT = 172.5, mW = 80.379, mN = 0, zero = 1e-12):
        sol = NuDoublePtEtaPhiE(         
                q1.pt/1000., q1.eta, q1.phi, q1.e/1000., 
                l1.pt/1000., l1.eta, l1.phi, l1.e/1000., 
                ev.met/1000., ev.met_phi, 
                S[0], S[1], S[2], S[3], mT, mW, mN, zero)
        skip = sol[0].tolist()[0]
        if skip: return []
        return [self.MakeNu(sol[1].tolist()[0])]

    def Sort(self, inpt, descending = False):
        if isinstance(inpt, list):
            inpt.sort()
            return inpt
        _tmp = list(inpt)
        _tmp.sort()
        if descending: _tmp.reverse()
        inpt = {k : inpt[k] for k in _tmp}
        return inpt
    
    def _EventPreprocessing(self, event):
        self.AllWeights += [event.weight]        
        if self.Selection(event) == False:
            if "Rejected::Selection" not in self.CutFlow: self.CutFlow["Rejected::Selection"] = 0
            self.CutFlow["Rejected::Selection"] += 1
            return False
        self._t1
        o = self.Strategy(event)
        self._t2
        
        if isinstance(o, str) and "->" in o:
            if o not in self.CutFlow: self.CutFlow[o] = 0
            self.CutFlow[o] += 1
        else: self.Residual += [o] if o != None else []
        self.SelWeights += [event.weight] 
        return True
            
    def __call__(self, Ana = None):
        if Ana == None: return self
        for i in Ana: self._EventPreprocessing(i)
    
    def __eq__(self, other):
        if other == None: return False
        return Code(other)._Hash == Code(self)._Hash
    
    def __radd__(self, other):
        if other == 0: return self
        return self.__add__(other)
    
    def __add__(self, other):
        keys = set(list(self.__dict__) + list(other.__dict__))
        for i in keys:
            if isinstance(self.__dict__[i], str): continue
            if i == "_CutFlow":
                k_ = set(list(self.__dict__[i]) + list(other.__dict__[i]))
                self.__dict__[i] |= {l : 0 for l in k_ if l not in self.__dict__[i]}
                other.__dict__[i] |= {l : 0 for l in k_ if l not in other.__dict__[i]}
            if i not in self.__dict__:
                self.__dict__[i] = other.__dict__[i]
                continue
            self.__dict__[i] = self.MergeData(self.__dict__[i], other.__dict__[i])
        
        out = SelectionTemplate()
        for i in self.__dict__: setattr(out, i, self.__dict__[i])  
        return out
    
    def RestoreSettings(self, inpt):
        for i in inpt: self.__dict__[i] = inpt[i]
        return self
            