from AnalysisG.Templates import SelectionTemplate

class TopKinematics(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)
        self.res_top_kinematics = {"pt" : [], "eta" : [], "phi" : [], "energy" : []}
        self.spec_top_kinematics = {"pt" : [], "eta" : [], "phi" : [], "energy" : []}
        self.deltaR = {"SS" : [], "RS" : [], "RR" : []}
        self.mass_combi = {"SS" : [], "RS" : [], "RR" : []}

    def Selection(self, event):
        tops = event.Tops
        if len([t for t in tops if t.FromRes == 1]) != 2: return False
        if len([t for t in tops if t.FromRes == 0]) != 2: return False
        return True

    def Strategy(self, event):
        tops = event.Tops

        rtops = [t for t in tops if t.FromRes == 1]
        self.res_top_kinematics["pt"]     += [r.pt/1000 for r in rtops]
        self.res_top_kinematics["eta"]    += [r.eta for r in rtops]
        self.res_top_kinematics["phi"]    += [r.phi for r in rtops]
        self.res_top_kinematics["energy"] += [r.e/1000 for r in rtops]

        stops = [t for t in tops if t.FromRes == 0]
        self.spec_top_kinematics["pt"]     += [r.pt/1000 for r in stops]
        self.spec_top_kinematics["eta"]    += [r.eta for r in stops]
        self.spec_top_kinematics["phi"]    += [r.phi for r in stops]
        self.spec_top_kinematics["energy"] += [r.e/1000 for r in stops]

        self.deltaR["RS"] += [r.DeltaR(s) for r in rtops for s in stops]
        self.deltaR["SS"] += [stops[0].DeltaR(stops[1])]
        self.deltaR["RR"] += [rtops[0].DeltaR(rtops[1])]

        self.mass_combi["RS"] += [(r + s).Mass/1000 for r in rtops for s in stops]
        self.mass_combi["SS"] += [(stops[0] + stops[1]).Mass/1000]
        self.mass_combi["RR"] += [(rtops[0] + rtops[1]).Mass/1000]

