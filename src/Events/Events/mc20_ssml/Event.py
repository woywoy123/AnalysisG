from AnalysisG.Events.Events.mc20_ssml.Particles import *
from AnalysisG.Templates import EventTemplate

from time import sleep

class SSML_MC20(EventTemplate):
    def __init__(self):
        EventTemplate.__init__(self)
        self.Objects = {
                "Tops" : Top(),
                "Jets" : Jet(),
                "Muons" : Muon(),
                "Electrons" : Electron(),
                "AbstractParticles" : AbstractParticle()
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
        self.Tops = {t.index : t for t in self.Tops.values()}
        self.AbstractParticles = {ap.index : ap for ap in self.AbstractParticles.values()}

        # link the abstract particles to the tops
        for ap in self.AbstractParticles.values():
            for i in ap.top_index:
                if i < 0: continue
                ap.Parent += [self.Tops[i]]
                self.Tops[i].Children += [ap]

        # see if any of the abstract particles match with the detector level objects.
        detector = []
        detector += list(self.Jets.values())
        detector += list(self.Muons.values())
        detector += list(self.Electrons.values())
        dr_map = {}
        pho_map = {}
        for d in detector:
            for ap in self.AbstractParticles.values():
                dr = d.DeltaR(ap)
                if d.Type == "jet" and ap.is_jet: dr_map[dr] = [d, ap]
                elif d.Type == "mu" and ap.is_lepton: dr_map[dr] = [d, ap]
                elif d.Type == "el" and ap.is_lepton: dr_map[dr] = [d, ap]
                else: pho_map[dr] = [d, ap]

        matched_ap = []
        # try to minimize the geometric distance between detector particles and abstract particle
        for dr in sorted(dr_map, reverse = False):
            det, ap = dr_map[dr]

            # make sure that only one abstract particle is matched to a detector object
            if ap in matched_ap: continue

            #link the objects
            det.Parent += [ap]
            ap.Children += [det]
            matched_ap += [ap]

        # make the attributes a list
        self.Tops = list(self.Tops.values())
        self.Particles = list(self.AbstractParticles.values())
        self.Detector = detector
