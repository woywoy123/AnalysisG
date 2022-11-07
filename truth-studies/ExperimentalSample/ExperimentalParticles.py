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
        Particle.__init__(self)

class Children(Particle):

    def __init__(self):
        self.Type = "children"
        self.index = self.Type + "_index"
        self.top_i = self.Type + "_TopIndex"
        self.jet_i = self.Type + "_JetIndex"
        self.truthjet_i = self.Type + "_TruthJetIndex"
        self.pdgid = self.Type + "_pdgid"
        Particle.__init__(self)


class Electron(Particle):

    def __init__(self):
        self.Type = "el"
        self.charge = self.Type + "_charge"
        Particle.__init__(self)

class Muon(Particle):
    
    def __init__(self):
        self.Type = "mu"
        self.charge = self.Type + "_charge"
        Particle.__init__(self)

class Jets(Particle):

    def __init__(self):
        self.Type = "jet"
        self.TopIndex = self.Type + "_TopIndex"
        Particle.__init__(self)


