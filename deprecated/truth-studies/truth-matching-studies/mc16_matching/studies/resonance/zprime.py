from AnalysisG.Templates import SelectionTemplate

class ZPrime(SelectionTemplate):

    def __init__(self):

        SelectionTemplate.__init__(self)
        self.zprime_mass_tops = {"mass" : [], "pt" : []}
        self.zprime_mass_children = {"mass" : [], "pt" : []}
        self.zprime_mass_truthjets = {"mass" : [], "pt" : []}
        self.zprime_mass_jets = {"mass" : [], "pt" : []}

    def Selection(self, event):
        # Check if the events have only two resonant tops
        tops = [t for t in event.Tops if t.FromRes == 1]
        if len(tops) == 2: return True
        return False

    def Strategy(self, event):
        rtops = [t for t in event.Tops if t.FromRes]
        tZp = sum(rtops)
        self.zprime_mass_tops["mass"] += [tZp.Mass/1000]
        self.zprime_mass_tops["pt"] += [tZp.pt/1000]

        t1, t2 = rtops
        cZp = sum(t1.Children + t2.Children)
        self.zprime_mass_children["mass"] += [cZp.Mass/1000]
        self.zprime_mass_children["pt"] += [cZp.pt/1000]

        tjZp = t1.TruthJets + t2.TruthJets
        if not len(tjZp): return False
        tjZp += [i for i in t1.Children if i.is_lep or i.is_nu]
        tjZp += [i for i in t2.Children if i.is_lep or i.is_nu]
        tjZp = sum(list(set(tjZp)))
        self.zprime_mass_truthjets["mass"] += [tjZp.Mass/1000]
        self.zprime_mass_truthjets["pt"] += [tjZp.pt/1000]


        jZp = t1.Jets + t2.Jets
        if not len(jZp): return False
        jZp += [i for i in t1.Children if i.is_nu]
        jZp += [i for i in t2.Children if i.is_nu]
        leps = event.Electrons + event.Muons
        for j in leps:
            if t1 not in j.Parent and t2 not in j.Parent: continue
            jZp += [j]

        jZp = sum(list(set(jZp)))
        self.zprime_mass_jets["mass"] += [jZp.Mass/1000]
        self.zprime_mass_jets["pt"] += [jZp.pt/1000]

