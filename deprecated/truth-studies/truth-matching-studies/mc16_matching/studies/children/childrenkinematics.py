from AnalysisG.Templates import SelectionTemplate

class ChildrenKinematics(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)
        self.res_kinematics  = {"pt" : [], "eta" : [], "phi" : [], "eta" : []}
        self.spec_kinematics = {"pt" : [], "eta" : [], "phi" : [], "eta" : []}

        self.res_pdgid_kinematics  = {}
        self.spec_pdgid_kinematics = {}

        self.res_decay_mode  = {"lep" : {}, "had" : {}}
        self.spec_decay_mode = {"lep" : {}, "had" : {}}

        self.mass_clustering = {
                "CTRR" : [], "FTRR" : [],"CTSS" : [], "FTSS" : [],"FTRS" : []
        }

        self.dr_clustering = {
                "CTRR" : [], "FTRR" : [],"CTSS" : [], "FTSS" : [],"FTRS" : []
        }

        self.top_pt_clustering = {
                "CTRR" : [], "FTRR" : [],"CTSS" : [], "FTSS" : [],"FTRS" : []
        }

        self.top_energy_clustering = {
                "CTRR" : [], "FTRR" : [],"CTSS" : [], "FTSS" : [],"FTRS" : []
        }

        self.top_children_dr = {
                "rlep" : [], "rhad" : [], "slep" : [], "shad" : []
        }

        self.fractional = {
                "rhad-pt" : {}, "rhad-energy" : {}, "shad-pt" : {}, "shad-energy" : {},
                "rlep-pt" : {}, "rlep-energy" : {}, "slep-pt" : {}, "slep-energy" : {},
        }


    def Selection(self, event):
        tops = event.Tops
        if len([t for t in tops if t.FromRes == 1]) != 2: return False
        if len([t for t in tops if t.FromRes == 0]) != 2: return False
        return True

    def DumpKinematics(self, dic, p):
        if "pt"     not in dic: dic["pt"] = []
        if "eta"    not in dic: dic["eta"] = []
        if "phi"    not in dic: dic["phi"] = []
        if "energy" not in dic: dic["energy"] = []

        dic["pt"]     += [p.pt/1000]
        dic["energy"] += [p.e/1000]
        dic["eta"]    += [p.eta]
        dic["phi"]    += [p.phi]

    def Strategy(self, event):
        res_t = [t for t in event.Tops if t.FromRes == 1]
        for x in sum([t.Children for t in res_t], []):
            self.DumpKinematics(self.res_kinematics, x)
            if x.symbol not in self.res_pdgid_kinematics:
                self.res_pdgid_kinematics[x.symbol] = {}
            self.DumpKinematics(self.res_pdgid_kinematics[x.symbol], x)

        for t in res_t:
            flag = "had"
            if len([k for k in t.Children if k.is_lep]): flag = "lep"
            for c in t.Children:
                self.DumpKinematics(self.res_decay_mode[flag], c)
                fg = "r" + flag
                self.top_children_dr[fg] += [t.DeltaR(c)]
                if c.symbol not in self.fractional[fg + "-pt"]:
                    self.fractional[fg + "-pt"][c.symbol] = []
                    self.fractional[fg + "-energy"][c.symbol] = []

                self.fractional[fg + "-pt"][c.symbol].append(c.pt/t.pt)
                self.fractional[fg + "-energy"][c.symbol].append(c.e/t.e)

        res_s = [t for t in event.Tops if t.FromRes == 0]
        for x in sum([t.Children for t in res_s], []):
            self.DumpKinematics(self.spec_kinematics, x)
            if x.symbol not in self.spec_pdgid_kinematics:
                self.spec_pdgid_kinematics[x.symbol] = {}
            self.DumpKinematics(self.spec_pdgid_kinematics[x.symbol], x)

        for t in res_s:
            flag = "had"
            if len([k for k in t.Children if k.is_lep]): flag = "lep"
            for c in t.Children:
                self.DumpKinematics(self.spec_decay_mode[flag], c)

                fg = "s" + flag
                self.top_children_dr[fg] += [t.DeltaR(c)]
                if c.symbol not in self.fractional[fg + "-pt"]:
                    self.fractional[fg + "-pt"][c.symbol] = []
                    self.fractional[fg + "-energy"][c.symbol] = []

                self.fractional[fg + "-pt"][c.symbol].append(c.pt/t.pt)
                self.fractional[fg + "-energy"][c.symbol].append(c.e/t.e)

        checked = []
        for p1 in [[t, c] for t in event.Tops for c in t.Children]:
            t1, c1 = p1
            for p2 in [[t, c] for t in event.Tops for c in t.Children]:
                t2, c2 = p2
                if c1 == c2 or c2 in checked: continue

                flag = ""
                if   t1.FromRes == 1 and t2.FromRes == 1: flag = "RR"
                elif t1.FromRes == 1 and t2.FromRes == 0: flag = "RS"
                elif t1.FromRes == 0 and t2.FromRes == 1: flag = "RS"
                elif t1.FromRes == 0 and t2.FromRes == 0: flag = "SS"

                if   t1 == t2: flag = "CT" + flag
                elif t1 != t2: flag = "FT" + flag

                self.dr_clustering[flag] += [c1.DeltaR(c2)]
                self.top_pt_clustering[flag] += [t2.pt/1000]
                self.top_energy_clustering[flag] += [t2.e/1000]
                self.mass_clustering[flag] += [sum(t1.Children + t2.Children).Mass/1000]

            checked.append(c1)

