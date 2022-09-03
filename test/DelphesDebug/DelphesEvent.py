from AnalysisTopGNN.Templates import EventTemplate 
from DelphesDebug.DelphesParticles import *



class Event(EventTemplate):
    def __init__(self):
        EventTemplate.__init__(self)
        self.runNumber = "Event.Number"
        self.Weight = "Event.Weight"
        self.Trees = ["Delphes"]
        self.Branches = ["Particle"]# "GenJet","Jet", "Electron", "Muon"]
        self.Objects = {
                "Particle" : Particle()
               # "GenJet" : GenJet(),
               # "Jet" : Jet(),
               # "Electron" : Electron(),
               # "Muon" : Muon(),
                #"Photon" : Photon(),
                #"MissingET" : MissingET()  
                      }
        self.DefineObjects()

    def CompileEvent(self, ClearVal):
        self.Final_Particles = []
        def Sort (a,target,Top_Number):
            if isinstance(a, list):
                for i in a:
                    target.Index = Top_Number
                    if i not in target.Decay:
                        target.Decay.append(i)
                    i.Index = Top_Number
                    Sort(i,target,Top_Number)
                return
            else:
                D1_i = a.Daughter1
                D2_i = a.Daughter2
                Stat = a.Status
                if Stat == 23:
                    a.Index = Top_Number
                    if a not in self.Final_Particles:
                        self.Final_Particles.append(a)
                elif Stat == 1:
                    a.Index = Top_Number
                    return             
                elif D1_i == D2_i:
                    if self.Particle[D1_i] not in a.Decay:
                        a.Decay.append(self.Particle[D1_i])
                    a.Index = Top_Number
                    return Sort(self.Particle[D1_i], a, Top_Number)
                elif D1_i != D2_i:
                    Decays = self.Particle[D1_i : (D2_i+1)]
                    a.Index = Top_Number
                    return Sort(Decays,a,Top_Number)
       
        self.CompileParticles(ClearVal)
        
        self.Tops = []
        self.Particle = self.DictToList(self.Particle)
        for e in range(len(self.Particle)):
            Target_Top = self.Particle[e]
            if abs(Target_Top.PID) == 6 and Target_Top.Status == 62:
                self.Tops.append(Target_Top)
                #print(self.Tops)
        
        self.Final_Particles = []
        for h in range(len(self.Tops)):
            Tops_Index = -1
            Anti_Tops_Index = -2
            if self.Tops[h].PID == 6:
               Top_Index = Tops_Index
            if self.Tops[h].PID == -6:
               Top_Index = Anti_Tops_Index
            Sort(self.Tops[h],self.Tops[h], Top_Index)
            
                 
        TMP = []
        All = []
        TMP += self.Tops
        TMP += self.Final_Particles 
        All += self.Particle
        #All += self.GenJet
       # All += self.MissingET
        
        for i in All:
            if i in TMP:
                ind = TMP.index(i)
                TMP.pop(ind)
                continue
            if len(TMP) == 0:
                break
            del i





 
