from AnalysisTopGNN.Templates import EventTemplate
from Particles import Jet, Top

class Event(EventTemplate):

    def __init__(self):
        EventTemplate.__init__(self)
        
        # ===== Event Variable Declaration ===== # 
        self.weight = "weight_mc"
        self.mu = "mu"
        self.met = "met_met"
        self.phi = "met_phi"

        self.Trees = ["nominal"]

        self.Objects = {
                "SomeJets" : Jet(), 
                "Tops" : Top()
                }
        # ===== End of declaration ===== #
        self.DefineObjects()
        self._CommitHash = "<Some Hash For Bookkeeping>"
    
    def CompileEvent(self): 
        # Create njets variable for the event 
        self.nJets = len(self.SomeJets) 
        
        # Do some particle truth matching 
        for i in self.SomeJets:
            index = self.SomeJets[i].MatchedTops
            if index[0] == -1:
                continue 
            for t in index:
                self.SomeJets[i].Parent.append(self.Tops[t])

        # (Optional) Convert the dictionary to particle list.
        self.SomeJets = self.DictToList(self.SomeJets)
        self.Tops = self.DictToList(self.Tops)
