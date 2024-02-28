from AnalysisG.Templates import SelectionTemplate

class DecayModes(SelectionTemplate):

    def __init__(self):

        SelectionTemplate.__init__(self)
        self.res_top_modes   = {"HH" : [], "HL" : [], "LL" : []}
        self.res_top_charges = {"SS" : [], "SO" : []}
        self.res_top_pdgid   = {}

        self.spec_top_modes  = {"HH" : [], "HL" : [], "LL" : []}
        self.spec_top_charges = {"SS" : [], "SO" : []}
        self.spec_top_pdgid  = {}

        # cases where one spectator and one resonant top decays leptonically
        self.signal_region = {"SS" : [], "SO" : []}
        self.ntops = []

        self.all_pdgid = {}

    def Selection(self, event):
        self.ntops += [len(event.Tops)]

        # Check if the events have only two resonant tops
        tops = [t for t in event.Tops if t.FromRes == 1]
        if len(tops) != 2: return False

        # Check if the events have only two spectator tops
        tops = [t for t in event.Tops if t.FromRes != 1]
        if len(tops) != 2: return False
        return True

    def Strategy(self, event):

        # resonant tops
        rtops = [t for t in event.Tops if t.FromRes == 1]
        mdicts = {"H" : 0, "L" : 0}
        cdicts = {"S" : 0, "O" : 0}
        for t in rtops:
            c = [i for i in t.Children if i.is_lep]
            if not len(c): mdicts["H"] += 1
            else: mdicts["L"] += 1

            if not len(c): continue
            if c[0].charge < 0: cdicts["S"] += 1
            else: cdicts["O"] += 1

        mheader = "H"*mdicts["H"] + "L"*mdicts["L"]
        cheader = ""
        if cdicts["S"] == 2 or cdicts["O"] == 2: cheader = "SS"
        elif cdicts["S"] == 1 or cdicts["O"] == 1: cheader = "SO"

        rchildren = [c for t in rtops for c in t.Children]
        res_c = sum(rchildren)
        self.res_top_modes[mheader] += [res_c.Mass/1000]
        if len(cheader): self.res_top_charges[cheader] += [res_c.Mass/1000]

        for c in rchildren:
            if c.symbol not in self.res_top_pdgid: self.res_top_pdgid[c.symbol] = 0
            if c.symbol not in self.all_pdgid: self.all_pdgid[c.symbol] = 0
            self.res_top_pdgid[c.symbol] += 1
            self.all_pdgid[c.symbol] += 1

        # spectator tops
        stops = [t for t in event.Tops if t.FromRes != 1]
        mdicts = {"H" : 0, "L" : 0}
        cdicts = {"S" : 0, "O" : 0}
        for t in stops:
            c = [i for i in t.Children if i.is_lep]
            if not len(c): mdicts["H"] += 1
            else: mdicts["L"] += 1

            if not len(c): continue
            if c[0].charge < 0: cdicts["S"] += 1
            else: cdicts["O"] += 1

        mheader = "H"*mdicts["H"] + "L"*mdicts["L"]
        cheader = ""
        if cdicts["S"] == 2 or cdicts["O"] == 2: cheader = "SS"
        elif cdicts["S"] == 1 or cdicts["O"] == 1: cheader = "SO"

        schildren = [c for t in stops for c in t.Children]
        res_s = sum(schildren)
        self.spec_top_modes[mheader] += [res_s.Mass/1000]
        if len(cheader): self.spec_top_charges[cheader] += [res_s.Mass/1000]

        for c in schildren:
            if c.symbol not in self.spec_top_pdgid: self.spec_top_pdgid[c.symbol] = 0
            if c.symbol not in self.all_pdgid: self.all_pdgid[c.symbol] = 0
            self.spec_top_pdgid[c.symbol] += 1
            self.all_pdgid[c.symbol] += 1

        signs = {"S" : 0, "O" : 0}
        for c in rchildren:
            if not c.is_lep: continue
            if c.charge < 0: signs["S"] += 1
            else: signs["O"] += 1

        for c in schildren:
            if not c.is_lep: continue
            if c.charge < 0: signs["S"] += 1
            else: signs["O"] += 1

        cheader = ""
        if signs["S"] == 2 or signs["O"] == 2: cheader = "SS"
        elif signs["S"] == 1 or signs["O"] == 1: cheader = "SO"
        else: return
        self.signal_region[cheader] += [res_c.Mass/1000]
