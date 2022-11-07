from AnalysisTopGNN.Templates import EventTemplate
from ExperimentalParticles import *

class EventExperimental(EventTemplate):

    def __init__(self):
        EventTemplate.__init__(self)
        self.Objects = {
                "Tops" : Top(), 
                "Children" : Children(),
        }
       
        self.Trees = ["nominal"]

        self.Lumi = "weight_mc"
        self.mu = "mu"

        self.DefineObjects()

    def CompileEvent(self):
        
        collect = {}
        for c in self.Children.values():
            if c.top_i not in collect:
                collect[c.top_i] = {}
            if c.index not in collect[c.top_i]:
                collect[c.top_i][c.index] = []
            collect[c.top_i][c.index].append(c)

            if c.index > 0:
                collect[c.top_i][c.index-1][-1].Children.append(c)
        
        for t in self.Tops.values():
            t.Children.append(collect[t.index][0][0])

        self.Tops = list(self.Tops.values())
