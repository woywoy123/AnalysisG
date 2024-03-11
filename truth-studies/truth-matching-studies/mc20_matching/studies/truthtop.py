from AnalysisG.Templates import SelectionTemplate

class TruthTops(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)
        self.event_ntops = []

        self.truth_top = []
        self.no_children = []

        self.truth_children = {"all" : [], "leptonic" : [], "hadronic" : []}
        self.truth_physics = {"all" : [], "leptonic" : [], "hadronic" : []}

        self.jets_truth_leps = {"all" : [], "leptonic" : [], "hadronic" : []}
        self.detector = {"all" : [], "leptonic" : [], "hadronic" : []}

        self.mtt_dr = {"dr" : [], "mass" : []}
        self.tops_kinematics = {"pt" : [], "eta" : [], "phi" : [], "energy" : []}
        self.tops_attributes = {"charge" : [], "barcode" : [], "status" : []}

    def populate_dict(self, maps, particle):
        for p in particle:
            maps["pt"].append(p.pt/1000)
            maps["energy"].append(p.e/1000)
            maps["eta"].append(p.eta)
            maps["phi"].append(p.phi)

    def Strategy(self, event):
        tops = event.Tops
        self.populate_dict(self.tops_kinematics, tops)

        self.tops_attributes["charge"] += [t.charge for t in tops]
        self.tops_attributes["barcode"] += [t.barcode for t in tops]
        self.tops_attributes["status"] += [t.status for t in tops]

        drs = [t for t in tops]
        for ti in tops:
            for tj in drs:
                if ti == tj: continue
                self.mtt_dr["dr"].append(ti.DeltaR(tj))
                self.mtt_dr["mass"].append(sum([ti + tj]).Mass/1000)
            drs = [t for t in drs if ti != t]
        self.event_ntops += [len(tops)]

        for t in tops:
            self.truth_top += [t.Mass/1000]
            mode = "hadronic"
            if len([i for i in t.Children if i.is_lep]): mode = "leptonic"
            c = sum(set(t.Children))

            try:
                self.truth_children["all"] += [c.Mass/1000]
                self.truth_children[mode]  += [c.Mass/1000]
            except AttributeError: pass

            tru_p  = [i for i in t.Children if i.is_nu]
            tru_p += [tr for tr in event.PhysicsTruth if t in tr.Parent and len(tr.Parent)]
            tru_p = sum(set(tru_p))
            try:
                self.truth_physics["all"] += [tru_p.Mass/1000]
                self.truth_physics[mode]  += [tru_p.Mass/1000]
            except AttributeError: pass
            continue

            tru_l  = [i for i in t.Children if i.is_lep or i.is_nu]
            tru_l += [i for i in event.Jets if t in i.Parent]
            tru_l = sum(set(tru_l))

            try:
                self.jets_truth_leps["all"] += [tru_l.Mass/1000]
                self.jets_truth_leps[mode]  += [tru_l.Mass/1000]
            except AttributeError: pass

            det_l  = event.Jets + event.Leptons
            det_l  = [i for i in det_l if t in i.Parent]
            det_l += [i for i in t.Children if i.is_nu]
            det_l  = sum(set(det_l))

            try:
                self.detector["all"] += [det_l.Mass/1000]
                self.detector[mode]  += [det_l.Mass/1000]
            except AttributeError: pass
