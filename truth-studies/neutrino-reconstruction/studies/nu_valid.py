import vector
from .nusol import singleNeutrinoSolution
from AnalysisG.Templates import SelectionTemplate


class NeutrinoReconstruction(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)
        self.children_kinematic_delta_pyc = {"px" : [], "py" : [], "pz" : [], "e" : []}
        self.children_kinematic_delta_ref = {"px" : [], "py" : [], "pz" : [], "e" : []}
        self.children_top_mass_diff = {"truth" : [], "pyc" : [], "reference" : []}

        self.truthjet_kinematic_delta_pyc = {"px" : [], "py" : [], "pz" : [], "e" : []}
        self.truthjet_kinematic_delta_ref = {"px" : [], "py" : [], "pz" : [], "e" : []}
        self.truthjet_top_mass_diff = {"truth" : [], "pyc" : [], "reference" : []}

        self.jet_kinematic_delta_pyc = {"px" : [], "py" : [], "pz" : [], "e" : []}
        self.jet_kinematic_delta_ref = {"px" : [], "py" : [], "pz" : [], "e" : []}
        self.jet_top_mass_diff = {"truth" : [], "pyc" : [], "reference" : []}

        self.reco_lep_jet_kinematic_delta_pyc = {"px" : [], "py" : [], "pz" : [], "e" : []}
        self.reco_lep_jet_kinematic_delta_ref = {"px" : [], "py" : [], "pz" : [], "e" : []}
        self.reco_lep_jet_top_mass_diff = {"truth" : [], "pyc" : [], "reference" : []}

        self.S_matrix_delta_bruteforce_pyc = {"px" : [], "py" : [], "pz" : [], "chi2" : [], "ii" : [], "ij" : []}
        self.S_matrix_delta_bruteforce_ref = {"px" : [], "py" : [], "pz" : [], "chi2" : [], "ii" : [], "ij" : []}

    # Select events where only one top-quark decays leptonically.
    def Selection(self, event):
        c = [1 for t in event.Tops for c in t.Children if c.is_lep]
        if len(c) == 1: return True
        return False

    def as_vec(self, inpt):
        dct = {"pt" : inpt.pt/1000, "eta" : inpt.eta, "phi" : inpt.phi, "energy" : inpt.e/1000}
        return vector.obj(**dct)

    def make_original(self, b, l, nu, met_x, met_y, diff_map, top_map):
        mW = (l + nu).Mass
        mT = (l + b + nu).Mass
        try:
            nu_ref = singleNeutrinoSolution(self.as_vec(b), self.as_vec(l), met_x/1000, met_y/1000, mW/1000, mT/1000)
            nu_ref = self.MakeNu(list(nu_ref.nu))
        except: return
        diff_map["px"] += [(nu_ref.px - nu.px)/1000]
        diff_map["py"] += [(nu_ref.py - nu.py)/1000]
        diff_map["pz"] += [(nu_ref.pz - nu.pz)/1000]
        diff_map["e"] += [(nu_ref.e - nu.e)/1000]
        top_map["reference"] += [(nu_ref + b + l).Mass/1000]

    def make_pyc(self, b, l, nu, event, diff_map, top_map):
        mW = (l + nu).Mass
        mT = (l + b + nu).Mass
        nu_pyc = self.Nu(b, l, event, S = [1000, 100, 100, 1000], mT = mT, mW = mW, zero = 1e-10)
        try: nu_pyc = nu_pyc[0]
        except IndexError: return
        except: return
        diff_map["px"] += [(nu_pyc.px - nu.px)/1000]
        diff_map["py"] += [(nu_pyc.py - nu.py)/1000]
        diff_map["pz"] += [(nu_pyc.pz - nu.pz)/1000]
        diff_map["e"] += [(nu_pyc.e - nu.e)/1000]
        top_map["pyc"] += [(nu_pyc + b + l).Mass/1000]

    def Strategy(self, event):
        t  = [t for t in event.Tops for c in t.Children if c.is_lep][0]
        l  = [c for c in t.Children if c.is_lep and not c.is_nu][0]
        nu = [c for c in t.Children if c.is_nu][0]
        b  = [c for c in t.Children if c.is_b][0]

        met = event.met
        phi = event.met_phi

        met_x = self.Px(met, phi)
        met_y = self.Py(met, phi)

        mW = (l + nu).Mass
        mT = (l + b + nu).Mass

        # // --------------------------- Truth Children Reconstruction --------------------------- //
        self.children_top_mass_diff["truth"].append(mT/1000)
        self.make_original(b, l, nu, met_x, met_y, self.children_kinematic_delta_ref, self.children_top_mass_diff)
        self.make_pyc(b, l, nu, event, self.children_kinematic_delta_pyc, self.children_top_mass_diff)

        # // ------------------------- Truth Jets + Child Lepton Reconstruction ------------------------- //
        for j in event.TruthJets:
            truthjet_b = None
            for p in j.Parton:
                if b not in p.Parent: continue
                truthjet_b = j
                break

            if truthjet_b is None: continue
            mT = (l + truthjet_b + nu).Mass
            self.truthjet_top_mass_diff["truth"].append(mT/1000)
            self.make_original(truthjet_b, l, nu, met_x, met_y, self.truthjet_kinematic_delta_ref, self.truthjet_top_mass_diff)
            self.make_pyc(truthjet_b, l, nu, event, self.truthjet_kinematic_delta_pyc, self.truthjet_top_mass_diff)
            break

        # // -------------------------- Jets + Child Lepton Reconstruction --------------------------- //
        for j in event.Jets:
            jet_b = None
            for p in j.Parton:
                if b not in p.Parent: continue
                jet_b = j
                break

            if jet_b is None: continue
            mT = (l + jet_b + nu).Mass
            self.jet_top_mass_diff["truth"].append(mT/1000)
            self.make_original(jet_b, l, nu, met_x, met_y, self.jet_kinematic_delta_ref, self.jet_top_mass_diff)
            self.make_pyc(jet_b, l, nu, event, self.jet_kinematic_delta_pyc, self.jet_top_mass_diff)
            break

        # // -------------------------- Jets + Reco Lepton Reconstruction --------------------------- //
        reco_lep = None
        for x in l.Children: reco_lep = x
        if reco_lep is None or jet_b is None: return
        mW = (reco_lep + nu).Mass
        mT = (reco_lep + jet_b + nu).Mass
        self.reco_lep_jet_top_mass_diff["truth"].append(mT/1000)
        self.make_original(jet_b, reco_lep, nu, met_x, met_y, self.reco_lep_jet_kinematic_delta_ref, self.reco_lep_jet_top_mass_diff)
        self.make_pyc(jet_b, reco_lep, nu, event, self.reco_lep_jet_kinematic_delta_pyc, self.reco_lep_jet_top_mass_diff)

        # // ------------------ Jets + Reco Lepton Reconstruction S-Matrix ------------------ //
        v_b, v_l = self.as_vec(jet_b), self.as_vec(reco_lep)
        step = 100
        ref = {"px" : -1, "py" : -1, "pz" : -1, "val" : -1, "ii" : -1, "ij" : -1}
        pyc = {"px" : -1, "py" : -1, "pz" : -1, "val" : -1, "ii" : -1, "ij" : -1}
        for ii in range(100):
            for ij in range(100):
                s_ii = ii*step + step
                s_ij = ij*step + step

                try:
                    nu_ref = singleNeutrinoSolution(v_b, v_l, met_x/1000, met_y/1000, mW/1000, mT/1000, S = [[s_ii, s_ij], [s_ij, s_ii]])
                    nu_ref = self.MakeNu(list(nu_ref.nu))
                except: nu_ref = None
                if nu_ref is not None:
                    dx, dy, dz = (nu_ref.px - nu.px)/1000, (nu_ref.py - nu.py)/1000, (nu_ref.pz - nu.pz)/1000
                    dl = pow(dx**2 + dy**2 + dz**2, 0.5)
                    if (ref["val"] > dl or ref["px"] == -1): ref = {"px" : dx, "py" : dy, "pz" : dz, "val" : dl, "ii" : s_ii, "ij": s_ij}

                try:
                    nu_pyc = self.Nu(jet_b, reco_lep, event, S = [s_ii, s_ij, s_ij, s_ii], mT = mT, mW = mW, zero = 1e-10)
                    nu_pyc = nu_pyc[0]
                except: nu_pyc = None
                if nu_pyc is not None:
                    dx, dy, dz = (nu_pyc.px - nu.px)/1000, (nu_pyc.py - nu.py)/1000, (nu_pyc.pz - nu.pz)/1000
                    dl = pow(dx**2 + dy**2 + dz**2, 0.5)
                    if (pyc["val"] > dl or pyc["px"] == -1): pyc = {"px" : dx, "py" : dy, "pz" : dz, "val" : dl, "ii" : s_ii, "ij" : s_ij}

        self.S_matrix_delta_bruteforce_pyc["px"] += [pyc["px"]]
        self.S_matrix_delta_bruteforce_pyc["py"] += [pyc["py"]]
        self.S_matrix_delta_bruteforce_pyc["pz"] += [pyc["pz"]]
        self.S_matrix_delta_bruteforce_pyc["ii"] += [pyc["ii"]]
        self.S_matrix_delta_bruteforce_pyc["ij"] += [pyc["ij"]]
        self.S_matrix_delta_bruteforce_pyc["chi2"] += [pyc["val"]]

        self.S_matrix_delta_bruteforce_ref["px"] += [ref["px"]]
        self.S_matrix_delta_bruteforce_ref["py"] += [ref["py"]]
        self.S_matrix_delta_bruteforce_ref["pz"] += [ref["pz"]]
        self.S_matrix_delta_bruteforce_ref["ii"] += [ref["ii"]]
        self.S_matrix_delta_bruteforce_ref["ij"] += [ref["ij"]]
        self.S_matrix_delta_bruteforce_ref["chi2"] += [ref["val"]]






