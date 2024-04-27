from AnalysisG.Events.Events.mc20_ssml.Particles import *
from AnalysisG.Templates import EventTemplate

class SSML_MC20(EventTemplate):
    def __init__(self):
        EventTemplate.__init__(self)
        self.Objects = {
                "Tops" : Top(),
                "TopChildren" : Children(),
                "PhysicsTruth" : PhysicsTruth(),
                "PhysicsDetectors" : PhysicsDetector(),

                "Jets" : Jet(),
                "Muons" : Muon(),
                "Electrons" : Electron(),
        }

        self.Trees = ["nominal_Loose"]
        self.is_mc = "mcChannelNumber"
        self.met_sum = "met_sumet"
        self.met = "met_met"
        self.phi = "met_phi"

        self.weight = "weight_mc"
        self.mu = "mu"

    def CompileEvent(self):
        # Properly assign the index to the associated object
        tops = {i.index : i for i in self.Tops.values()}
        for c in self.TopChildren.values():
            c.Parent += [tops[c.index]]
            tops[c.index].Children += [c]

        for c in self.PhysicsTruth.values():
            c.Parent += [tops[x] for x in c.top_index if x > -1]

        for c in self.PhysicsDetectors.values():
            c.Parent += [tops[x] for x in c.top_index if x > -1]

        detectors  = list(self.Jets.values())
        detectors += list(self.Muons.values())
        detectors += list(self.Electrons.values())
        for d in detectors:
            maps = {d.DeltaR(pd) : pd for pd in self.PhysicsDetectors.values()}
            maps = dict(sorted(maps.items()))
            if list(maps)[0] > 0.0001: continue
            d.Parent += list(maps.values())[0].Parent

        # make the attributes a list
        self.Tops          = list(self.Tops.values())
        self.TruthChildren = list(self.TopChildren.values())
        self.PhysicsTruth  = list(self.PhysicsTruth.values())

        self.Jets            = list(self.Jets.values())
        self.Leptons         = list(self.Muons.values()) + list(self.Electrons.values())
        self.PhysicsDetector = list(self.PhysicsDetectors.values())
        self.Detector        = detectors
