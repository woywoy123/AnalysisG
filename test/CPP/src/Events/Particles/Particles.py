from AnalysisG.Templates import ParticleTemplate

class Particle(ParticleTemplate):

    def __init__(self):
        ParticleTemplate.__init__(self)

        self.pt = self.Type + "_pt"
        self.eta = self.Type + "_eta"
        self.phi = self.Type + "_phi"
        self.e = self.Type + "_e"

class Top(Particle):

    def __init__(self):
        self.Type = "top"
        Particle.__init__(self)
        self.index = self.Type + "_index"
        self.charge = self.Type + "_charge"
        self.pdgid = self.Type + "_pdgid"
        self.FromRes = self.Type + "_FromRes"
        self.status = self.Type + "_status"
        self.TruthJets = []
        self.Jets = []

class Children(Particle):

    def __init__(self):
        self.Type = "children"
        Particle.__init__(self)

        self.index = self.Type + "_index"
        self.charge = self.Type + "_charge"
        self.pdgid = self.Type + "_pdgid"
        self.TopIndex = self.Type + "_TopIndex"

    @property
    def FromRes(self):
        return self.Parent[0].FromRes

class TruthJet(Particle):
    def __init__(self):
        self.Type = "truthjets"
        Particle.__init__(self)

        self.index = self.Type + "_index"
        self.is_b_var = self.Type + "_btagged"
        self.TopQuarkCount = self.Type + "_topquarkcount"
        self.WBosonCount = self.Type + "_wbosoncount"
        self.TopIndex = self.Type + "_TopIndex"
        self.Tops = []
        self.Parton = []

    @property
    def FromRes(self):
        return 0 if len(self.Tops) == 0 else self.Tops[0].FromRes


class TruthJetParton(Particle):

    def __init__(self):
        self.Type = "TJparton"
        Particle.__init__(self)

        self.index = self.Type + "_index"
        self.TruthJetIndex = self.Type + "_TruthJetIndex"
        self.TopChildIndex = self.Type + "_ChildIndex"
        self.charge = self.Type + "_charge"
        self.pdgid = self.Type + "_pdgid"

class Jet(Particle):

    def __init__(self):
        self.Type = "jet"
        Particle.__init__(self)

        self.index = self.Type + "_index"
        self.TopIndex = self.Type + "_TopIndex"
        self.Tops = []
        self.btagged = self.Type + "_isbtagged_DL1r_77"
        self.Parton = []

    @property
    def is_b(self):
        return self.btagged

    @property
    def FromRes(self):
        if len(self.Tops) == 0:
            return False
        return self.Tops[0].FromRes

class JetParton(Particle):

    def __init__(self):
        self.Type = "Jparton"
        Particle.__init__(self)

        self.index = self.Type + "_index"
        self.JetIndex = self.Type + "_JetIndex"
        self.TopChildIndex = self.Type + "_ChildIndex"
        self.charge = self.Type + "_charge"
        self.pdgid = self.Type + "_pdgid"

class Electron(Particle):

    def __init__(self):
        self.Type = "el"
        Particle.__init__(self)
        self.charge = self.Type + "_charge"
        self.index = []

class Muon(Particle):

    def __init__(self):
        self.Type = "mu"
        Particle.__init__(self)
        self.charge = self.Type + "_charge"
        self.index = []

class Neutrino(ParticleTemplate):
    def __init__(self, px=None, py=None, pz=None):
        self.Type = "nu"
        ParticleTemplate.__init__(self)
        self.px = px
        self.py = py
        self.pz = pz