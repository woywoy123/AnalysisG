
class Particle:
    def __init__(self):
        self.pt = self.Type + "_pt"
        self.eta = self.Type + "_eta"
        self.phi = self.Type + "_phi"
        self.e = self.Type + "_e"
        self.Index = -1
        self.Signal = None
        self.Branches = []
        for i in list(self.__dict__.values()):
            if isinstance(i, str) and self.Type != i:
                self.Branches.append(i)

class TruthJet(Particle):
    def __init__(self):
        self.Type = "truthjet"
        Particle.__init__(self)

class Jet(Particle):
    def __init__(self):
        self.Type = "jet"
        Particle.__init__(self)

class Electron(Particle):
    def __init__(self):
        self.Type = "el"
        Particle.__init__(self)

class Muon(Particle):
    def __init__(self):
        self.Type = "el"
        Particle.__init__(self)

class Top(Particle):
    def __init__(self):
        self.Type = "truth_top"
        Particle.__init__(self)
        self.FromRes = "_FromRes"

class Truth_Top_Child(Particle):
    def __init__(self):
        self.Type = "truth_top_child"
        self.pdgid = self.Type + "_pdgid"
        Particle.__init__(self)

class Truth_Top_Child_Init(Particle):
    def __init__(self):
        self.Type = "truth_top_initialState_child"
        self.pdgid = "top_initialState_child_pdgid"
        Particle.__init__(self)




