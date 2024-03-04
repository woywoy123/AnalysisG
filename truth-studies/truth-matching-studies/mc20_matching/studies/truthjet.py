from AnalysisG.Templates import SelectionTemplate

class TruthJetMatching(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)
        self.abstract_top = {}
        self.decaymode = {"hadronic" : [], "leptonic" : []}
        self.flavor = {"hadronic" : [], "leptonic" : []}

    def Strategy(self, event):
        tops = event.Tops
        for t in tops:
            if t.index not in self.abstract_top: self.abstract_top[t.index] = []
            if not len(t.Children):
                self.abstract_top[t.index] += [0]
                continue

            self.abstract_top[t.index] += [sum(set(t.Children)).Mass/1000]

            print("")
            print([i.is_lepton for i in t.Children])



            lep = len([i for i in t.Children if i.is_lepton])
            if lep: self.decaymode["leptonic"] += [sum(set(t.Children)).Mass/1000]
            else: self.decaymode["hadronic"] += [sum(set(t.Children)).Mass/1000]

            typx = [i.true_flavor for i in t.Children]
            if lep: self.flavor["leptonic"] += typx
            else: self.flavor["hadronic"] += typx
