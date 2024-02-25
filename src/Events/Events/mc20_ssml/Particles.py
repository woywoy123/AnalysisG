from AnalysisG.Templates import ParticleTemplate


class Particle(ParticleTemplate):
    def __init__(self):
        ParticleTemplate.__init__(self)

        self.pt = self.Type + "_pt"
        self.eta = self.Type + "_eta"
        self.phi = self.Type + "_phi"
        self.e = self.Type + "_e"

    @property
    def is_lepton(self):
        if self.Type == "el": return True
        if self.Type == "mu": return True
        return False


class Electron(Particle):

    def __init__(self):
        self.Type = "el"
        Particle.__init__(self)
        self.charge = "el_charge"
        self.tight = "el_isTight"
        self.d0sig = "el_d0sig"
        self.delta_z0 = "el_delta_z0_sintheta"
        self.true_type = "el_true_type"
        self.true_origin = "el_true_origin"


class Muon(Particle):

    def __init__(self):
        self.Type = "mu"
        Particle.__init__(self)
        self.charge = "mu_charge"
        self.tight = "mu_isTight"
        self.d0sig = "mu_d0sig"
        self.delta_z0 = "mu_delta_z0_sintheta"
        self.true_type = "mu_true_type"
        self.true_origin = "mu_true_origin"

class Jet(Particle):

    def __init__(self):
        self.Type = "jet"
        Particle.__init__(self)
        self.jvt = "jet_jvt"
        self.truth_flavor = "jet_truthflav"
        self.truth_parton = "jet_truthPartonLabel"

        _tag = "jet_isbtagged_"
        self.btag60 = _tag + "GN2v00NewAliasWP_60"
        self.btag70 = _tag + "GN2v00NewAliasWP_70"
        self.btag77 = _tag + "GN2v00NewAliasWP_77"
        self.btag85 = _tag + "GN2v00NewAliasWP_85"


class AbstractParticle(Particle):

    def __init__(self):
        self.Type = "object"
        Particle.__init__(self)
        self.index = "object_index"
        self.charge = "object_charge"

        self.particle_type = "object_type"
        self.top_index = "object_top_index"
        self.true_flavor = "object_trueflavor"

        self.truth_cone = "object_conetruthlabel"
        self.truth_label = "object_partontruthlabel"

    @property
    def is_jet(self): return self.particle_type[0] == 1

    @property
    def is_lepton(self): return self.particle_type[1] == 1

    @property
    def is_photon(self): return self.particle_type[2] == 1

class Top(Particle):

    def __init__(self):
        self.Type = "top"
        Particle.__init__(self)
        self.index = "top_top_index"
        self.barcode = "top_barcode"
        self.charge = "top_charge"
        self.pdgid  = "top_pdgid"
        self.status = "top_status"




