from AnalysisTopGNN.Templates import ParticleTemplate

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
        self.index = self.Type + "_index"
        self.charge = self.Type + "_charge"
        self.pdgid = self.Type + "_pdgid"
        self.FromRes = self.Type + "_FromRes"
        Particle.__init__(self)
        self.TruthJets = []
        self.Jets = []

class Children(Particle):

    def __init__(self):
        self.Type = "children"
        self.charge = self.Type + "_charge"
        self.pdgid = self.Type + "_pdgid"
        self.index = self.Type + "_TopIndex"
        Particle.__init__(self)
        self.TruthJetPartons = []
        self.JetPartons = []

class TruthJet(Particle):

    def __init__(self):
        self.Type = "truthjet"
        self.index = self.Type + "_TopIndex"
        Particle.__init__(self)
        self.Partons = []

class TruthJetPartons(Particle):

    def __init__(self):
        self.Type = "truJparton"
        self.charge = self.Type + "_charge"
        self.pdgid = self.Type + "_pdgid"
        self.index = self.Type + "_ChildIndex"
        self.TruJetIndex = self.Type + "_TruJetIndex"
        Particle.__init__(self)
        self.TruthJet = []

class Electron(Particle):

    def __init__(self):
        self.Type = "el"
        self.charge = self.Type + "_charge"
        Particle.__init__(self)
        self.index = []

class Muon(Particle):
    
    def __init__(self):
        self.Type = "mu"
        self.charge = self.Type + "_charge"
        Particle.__init__(self)
        self.index = []

class Jets(Particle):

    def __init__(self):
        self.Type = "jet"
        self.index = self.Type + "_TopIndex"
        Particle.__init__(self)
        self.Partons = []

class JetPartons(Particle):

    def __init__(self):
        self.Type = "Jparton"
        self.charge = self.Type + "_charge"
        self.pdgid = self.Type + "_pdgid"
        self.index = self.Type + "_ChildIndex"
        self.JetIndex = self.Type + "_JetIndex"
        Particle.__init__(self)
        self.Jet = []


