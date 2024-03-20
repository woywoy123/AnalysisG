from AnalysisG.Templates import SelectionTemplate

class TopTruthJets(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)

        self.top_mass = {
                "hadronic" : [], "leptonic" : [],
                "ntruthjets" : {"hadronic" : {}, "leptonic" : {}},
                "merged_tops" : {"hadronic" : {}, "leptonic" : {}}
        }

        self.ntops_lost = []

    def Selection(self, event): return True

    def Strategy(self, event):
        lost = 0
        for t in event.Tops:
            frac = []
            frac += t.TruthJets
            if not len(frac): lost += 1; continue
            frac += [c for c in t.Children if c.is_lep or c.is_nu]
            top_mass = sum(frac).Mass/1000

            n_truj = len(t.TruthJets)
            n_tops = len(set([t_ for tj in t.TruthJets for t_ in tj.Tops]))

            is_lep = len([c for c in t.Children if c.is_lep]) != 0
            mode = "leptonic" if is_lep else "hadronic"
            self.top_mass[mode] += [top_mass]
            if n_truj not in self.top_mass["ntruthjets"][mode]:
                self.top_mass["ntruthjets"][mode][n_truj] = []
            self.top_mass["ntruthjets"][mode][n_truj] += [top_mass]

            if n_tops not in self.top_mass["merged_tops"][mode]:
                self.top_mass["merged_tops"][mode][n_tops] = []
            self.top_mass["merged_tops"][mode][n_tops] += [top_mass]
        self.ntops_lost += [lost]

