from AnalysisG.Templates import SelectionTemplate

class TruthTops(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)

        self.ntops = []

        self.tops_mass = {"all" : []}
        self.mtt_dr = {"dr" : [], "mass" : []}

        self.tops_kinematics = {"pt" : [], "eta" : [], "phi" : [], "energy" : []}
        self.tops_attributes = {"charge" : [], "barcode" : [], "status" : []}
        self.event_ntops = []

    def populate_dict(self, maps, particle):
        for p in particle:
            maps["pt"].append(p.pt/1000)
            maps["energy"].append(p.e/1000)
            maps["eta"].append(p.eta)
            maps["phi"].append(p.phi)

    def Strategy(self, event):
        tops = event.Tops
        self.tops_mass["all"] += [t.Mass/1000 for t in tops]
        self.populate_dict(self.tops_kinematics, tops)

        self.tops_attributes["charge"] += [t.charge for t in tops]
        self.tops_attributes["barcode"] += [t.barcode for t in tops]
        self.tops_attributes["status"] += [t.status for t in tops]
        self.ntops += [len(tops)]

        drs = [t for t in tops]
        for ti in tops:
            for tj in drs:
                if ti == tj: continue
                self.mtt_dr["dr"].append(ti.DeltaR(tj))
                self.mtt_dr["mass"].append(sum([ti + tj]).Mass/1000)
            drs = [t for t in drs if ti != t]
        self.event_ntops += [len(tops)]

