from AnalysisG.Templates import SelectionTemplate
from .nusol import doubleNeutrinoSolutions

class DoubleNeutrinoReconstruction(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)


        self.children_kinematic_delta_ref = {
                "nu1" : {"px" : [], "py" : [], "pz" : [], "e" : []},
                "nu2" : {"px" : [], "py" : [], "pz" : [], "e" : []},
        }

        self.children_kinematic_delta_pyc = {
                "nu1" : {"px" : [], "py" : [], "pz" : [], "e" : []},
                "nu2" : {"px" : [], "py" : [], "pz" : [], "e" : []}
        }

        self.children_top_mass_diff = {
                "nu1" : {"truth" : [], "pyc" : [], "reference" : [], "reference-no-optim" : [], "reference-optim" : []},
                "nu2" : {"truth" : [], "pyc" : [], "reference" : [], "reference-no-optim" : [], "reference-optim" : []},
        }

        self.truthjet_kinematic_delta_ref = {
                "nu1" : {"px" : [], "py" : [], "pz" : [], "e" : []},
                "nu2" : {"px" : [], "py" : [], "pz" : [], "e" : []}
        }

        self.truthjet_kinematic_delta_pyc = {
                "nu1" : {"px" : [], "py" : [], "pz" : [], "e" : []},
                "nu2" : {"px" : [], "py" : [], "pz" : [], "e" : []},
        }

        self.truthjet_top_mass_diff = {
                "nu1" : {"truth" : [], "pyc" : [], "reference" : [], "reference-no-optim" : [], "reference-optim" : []},
                "nu2" : {"truth" : [], "pyc" : [], "reference" : [], "reference-no-optim" : [], "reference-optim" : []}
        }

        self.jet_kinematic_delta_pyc = {
                "nu1" : {"px" : [], "py" : [], "pz" : [], "e" : []},
                "nu2" : {"px" : [], "py" : [], "pz" : [], "e" : []},
        }

        self.jet_kinematic_delta_ref = {
                "nu1" : {"px" : [], "py" : [], "pz" : [], "e" : []},
                "nu2" : {"px" : [], "py" : [], "pz" : [], "e" : []}
        }

        self.jet_top_mass_diff = {
                "nu1" : {"truth" : [], "pyc" : [], "reference" : [], "reference-no-optim" : [], "reference-optim" : []},
                "nu2" : {"truth" : [], "pyc" : [], "reference" : [], "reference-no-optim" : [], "reference-optim" : []}
        }

        self.reco_lep_jet_kinematic_delta_pyc = {
                "nu1" : {"px" : [], "py" : [], "pz" : [], "e" : []},
                "nu2" : {"px" : [], "py" : [], "pz" : [], "e" : []}
        }

        self.reco_lep_jet_kinematic_delta_ref = {
                "nu1" : {"px" : [], "py" : [], "pz" : [], "e" : []},
                "nu2" : {"px" : [], "py" : [], "pz" : [], "e" : []}
        }

        self.reco_lep_jet_top_mass_diff = {
                "nu1" : {"truth" : [], "pyc" : [], "reference" : [], "reference-no-optim" : [], "reference-optim" : []},
                "nu2" : {"truth" : [], "pyc" : [], "reference" : [], "reference-no-optim" : [], "reference-optim" : []}
        }

    def Selection(self, event):
        res = 0
        for t in event.Tops:
            x = [1 for c in t.Children if c.is_lep]
            if not len(x): continue
            res += 1
        if res == 2: return True
        return False

    def as_vec(self, inpt):
        dct = {"pt" : inpt.pt, "eta" : inpt.eta, "phi" : inpt.phi, "energy" : inpt.e}
        return vector.obj(**dct)

    def make_original(self, b1, b2, l1, l2, nu1, nu2, met_x, met_y, diff_map, top_map):
        mW = ((l1 + nu1).Mass + (l2 + nu2).Mass)/2
        mT = ((l1 + b1 + nu1).Mass + (l2 + b2 + nu2).Mass)/2
        bs = (self.as_vec(b1), self.as_vec(b2))
        ls = (self.as_vec(l1), self.as_vec(l2))
        optims = False
        try:
            nu_ref = doubleNeutrinoSolutions(bs, ls, met_x, met_y, mW, mT)
            optims = nu_ref.opt
        except: return
        nu_ref = nu_ref.nunu_s
        if not len(nu_ref): return
        for sols in nu_ref:
            nu_ref = [self.MakeNu(sols[0]), self.MakeNu(sols[1])]
            break
        diff_map["nu1"]["px"] += [(nu_ref[0].px - nu1.px)/1000]
        diff_map["nu1"]["py"] += [(nu_ref[0].py - nu1.py)/1000]
        diff_map["nu1"]["pz"] += [(nu_ref[0].pz - nu1.pz)/1000]
        diff_map["nu1"]["e"]  += [(nu_ref[0].e  - nu1.e )/1000]

        diff_map["nu2"]["px"] += [(nu_ref[1].px - nu2.px)/1000]
        diff_map["nu2"]["py"] += [(nu_ref[1].py - nu2.py)/1000]
        diff_map["nu2"]["pz"] += [(nu_ref[1].pz - nu2.pz)/1000]
        diff_map["nu2"]["e"]  += [(nu_ref[1].e  - nu2.e )/1000]

        mt1 = (nu_ref[0] + b1 + l1).Mass/1000
        mt2 = (nu_ref[1] + b2 + l2).Mass/1000
        top_map["nu1"]["reference"] += [mt1]
        top_map["nu2"]["reference"] += [mt2]

        key = "reference"
        if optims: key += "-optim"
        else: key += "-no-optim"
        top_map["nu1"][key] += [mt1]
        top_map["nu2"][key] += [mt2]

    def make_pyc(self, b1, b2, l1, l2, nu1, nu2, event, diff_map, top_map):
        mW = ((l1 + nu1).Mass + (l2 + nu2).Mass)/2
        mT = ((l1 + b1 + nu1).Mass + (l2 + b2 + nu2).Mass)/2
        nunu_pyc = self.NuNu(b1, b2, l1, l2, event, mT = mT, mW = mW, zero = 1e-10)
        if not len(nunu_pyc): return
        nunu_pyc = nunu_pyc[0]

        diff_map["nu1"]["px"] += [(nunu_pyc[0].px - nu1.px)/1000]
        diff_map["nu1"]["py"] += [(nunu_pyc[0].py - nu1.py)/1000]
        diff_map["nu1"]["pz"] += [(nunu_pyc[0].pz - nu1.pz)/1000]
        diff_map["nu1"]["e"]  += [(nunu_pyc[0].e  - nu1.e )/1000]

        diff_map["nu2"]["px"] += [(nunu_pyc[1].px - nu2.px)/1000]
        diff_map["nu2"]["py"] += [(nunu_pyc[1].py - nu2.py)/1000]
        diff_map["nu2"]["pz"] += [(nunu_pyc[1].pz - nu2.pz)/1000]
        diff_map["nu2"]["e"]  += [(nunu_pyc[1].e  - nu2.e )/1000]

        top_map["nu1"]["pyc"] += [(nunu_pyc[0] + b1 + l1).Mass/1000]
        top_map["nu2"]["pyc"] += [(nunu_pyc[1] + b2 + l2).Mass/1000]

    def Strategy(self, event):
        lep_tops = []
        for t in event.Tops:
            x = [1 for c in t.Children if c.is_lep]
            if not len(x): continue
            lep_tops.append(t)
        t1, t2 = lep_tops
        b1  = [i for i in t1.Children if i.is_b][0]
        l1  = [i for i in t1.Children if i.is_lep and not i.is_nu][0]
        nu1 = [i for i in t1.Children if i.is_nu][0]

        b2  = [i for i in t2.Children if i.is_b][0]
        l2  = [i for i in t2.Children if i.is_lep and not i.is_nu][0]
        nu2 = [i for i in t2.Children if i.is_nu][0]

        met = event.met
        phi = event.met_phi

        met_x = self.Px(met, phi)
        met_y = self.Py(met, phi)

        mW = ((l1 + nu1).Mass + (l2 + nu2).Mass)/2
        mT = ((l1 + b1 + nu1).Mass + (l2 + b2 + nu2).Mass)/2

        self.make_original(b1, b2, l1, l2, nu1, nu2, met_x, met_y, self.children_kinematic_delta_ref, self.children_top_mass_diff)
        self.make_pyc(b1, b2, l1, l2, nu1, nu2, event, self.children_kinematic_delta_pyc, self.children_top_mass_diff)

        self.children_top_mass_diff["nu1"]["truth"] += [(nu1 + l1 + b1).Mass/1000]
        self.children_top_mass_diff["nu2"]["truth"] += [(nu2 + l2 + b2).Mass/1000]

        tjb1 = None
        for j in t1.TruthJets:
            # assert that a truth jet has no more than 2 top contributions
            if len(j.Tops) > 1: continue
            for p in j.Parton:
                if b1 not in p.Parent: continue
                tjb1 = j
            if tjb1 is None: continue
            break

        tjb2 = None
        for j in t2.TruthJets:
            # assert that a truth jet has no more than 2 top contributions
            if len(j.Tops) > 1: continue
            for p in j.Parton:
                if b2 not in p.Parent: continue
                tjb2 = j
            if tjb2 is None: continue
            break

        if tjb1 is not None and tjb2 is not None:
            self.make_original(tjb1, tjb2, l1, l2, nu1, nu2, met_x, met_y, self.truthjet_kinematic_delta_ref, self.truthjet_top_mass_diff)
            self.make_pyc(tjb1, tjb2, l1, l2, nu1, nu2, event, self.truthjet_kinematic_delta_pyc, self.truthjet_top_mass_diff)

            self.truthjet_top_mass_diff["nu1"]["truth"] += [(nu1 + l1 + tjb1).Mass/1000]
            self.truthjet_top_mass_diff["nu2"]["truth"] += [(nu2 + l2 + tjb2).Mass/1000]

        jb1 = None
        for j in t1.Jets:
            # assert that a truth jet has no more than 2 top contributions
            if len(j.Tops) > 1: continue
            for p in j.Parton:
                if b1 not in p.Parent: continue
                jb1 = j
            if jb1 is None: continue
            break

        jb2 = None
        for j in t2.Jets:
            # assert that a truth jet has no more than 2 top contributions
            if len(j.Tops) > 1: continue
            for p in j.Parton:
                if b2 not in p.Parent: continue
                jb2 = j
            if jb2 is None: continue
            break

        if jb1 is not None and jb2 is not None:
            self.make_original(jb1, jb2, l1, l2, nu1, nu2, met_x, met_y, self.jet_kinematic_delta_ref, self.jet_top_mass_diff)
            self.make_pyc(jb1, jb2, l1, l2, nu1, nu2, event, self.jet_kinematic_delta_pyc, self.jet_top_mass_diff)

            self.jet_top_mass_diff["nu1"]["truth"] += [(nu1 + l1 + jb1).Mass/1000]
            self.jet_top_mass_diff["nu2"]["truth"] += [(nu2 + l2 + jb2).Mass/1000]

        if jb1 is None or jb2 is None: return
        lep1, lep2 = l1.Children, l2.Children
        if not len(lep1): return
        if not len(lep2): return
        lep1, lep2 = lep1[0], lep2[0]

        self.make_original(jb1, jb2, lep1, lep2, nu1, nu2, met_x, met_y, self.reco_lep_jet_kinematic_delta_ref, self.reco_lep_jet_top_mass_diff)
        self.make_pyc(jb1, jb2, lep1, lep2, nu1, nu2, event, self.reco_lep_jet_kinematic_delta_pyc, self.reco_lep_jet_top_mass_diff)

        self.reco_lep_jet_top_mass_diff["nu1"]["truth"] += [(nu1 + lep1 + jb1).Mass/1000]
        self.reco_lep_jet_top_mass_diff["nu2"]["truth"] += [(nu2 + lep2 + jb2).Mass/1000]







