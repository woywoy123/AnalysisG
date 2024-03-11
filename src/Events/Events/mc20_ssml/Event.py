from AnalysisG.Events.Events.mc20_ssml.Particles import *
from AnalysisG.Templates import EventTemplate

from time import sleep

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
            for x in c.top_index:
                if x < 0: continue
                c.Parent += [self.Tops[x]]
            c.Parent = list(set(c.Parent))

        for c in self.PhysicsDetectors.values():
            for x in c.top_index:
                if x < 0: continue
                c.Parent += [self.Tops[x]]

        for d1 in list(self.PhysicsTruth.values()):
            dr_map = {d1.DeltaR(d2) : [d1, d2] for d2 in self.PhysicsDetectors.values()}
            try: c1, c2 = dr_map[0]
            except KeyError: continue
            c1.Children += [c2]

        detectors = list(self.Jets.values())
        detectors += list(self.Muons.values())
        detectors += list(self.Electrons.values())
        for d in detectors:
            for pd in self.PhysicsDetectors.values():
                if d != pd: continue
                d.Parent += [k for k in pd.Parent]

        # make the attributes a list
        self.Tops          = list(self.Tops.values())
        self.TruthChildren = list(self.TopChildren.values())
        self.PhysicsTruth  = list(self.PhysicsTruth.values())

        self.Jets          = list(self.Jets.values())
        self.Leptons       = list(self.Muons.values()) + list(self.Electrons.values())
        self.Detector      = list(self.PhysicsDetectors.values())
