from AnalysisG.Templates import SelectionTemplate

class TopMatching(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)
        self.truth_top = []

        self.truth_children = {"all" : [], "lep" : [], "had" : []}
        self.truth_jets = {"all" : [], "lep" : [], "had" : []}

        self.jets_truth_leps = {"all" : [], "lep" : [], "had" : []}
        self.jets_leps = {"all" : [], "lep" : [], "had" : []}


        self.n_truth_jets_lep = {}
        self.n_truth_jets_had = {}

        self.n_jets_lep = {}
        self.n_jets_had = {}

        self.no_children = []

    def Selection(self, event): return True

    def Strategy(self, event):
        tops = event.Tops
        for t in tops: self.truth_top += [t.Mass/1000]
        dleps = event.Electrons + event.Muons

        for t in tops:
            ch = t.Children
            if not len(ch): self.no_children += [1]

            ch_ = [c for c in ch if c.is_lep]
            if len(ch):
                self.truth_children["all"] += [sum(ch).Mass/1000]
                if len(ch_): self.truth_children["lep"] += [sum(ch).Mass/1000]
                else: self.truth_children["had"] += [sum(ch).Mass/1000]

            tj = t.TruthJets + [c for c in ch if c.is_lep or c.is_nu]
            if len(t.TruthJets):
                mass = sum(tj).Mass/1000
                if len(ch_): self.truth_jets["lep"] += [mass]
                else: self.truth_jets["had"] += [mass]
                self.truth_jets["all"] += [mass]

                # Find the number of truth jet contributions 
                ntj = str(len(t.TruthJets)) + " - Truth Jets"
                if len(ch_):
                    if ntj not in self.n_truth_jets_lep: self.n_truth_jets_lep[ntj] = []
                    self.n_truth_jets_lep[ntj] += [mass]
                else:
                    if ntj not in self.n_truth_jets_had: self.n_truth_jets_had[ntj] = []
                    self.n_truth_jets_had[ntj] += [mass]

            if not len(t.Jets): continue

            jt = t.Jets + [c for c in ch if c.is_lep or c.is_nu]
            mass = sum(jt).Mass/1000
            if len(ch_): self.jets_truth_leps["lep"] += [mass]
            else: self.jets_truth_leps["had"] += [mass]
            self.jets_truth_leps["all"] += [mass]

            jts = t.Jets + [l for l in dleps if t in l.Parent] + [c for c in ch if c.is_nu]
            mass = sum(jts).Mass/1000
            if len(ch_): self.jets_leps["lep"] += [mass]
            else: self.jets_leps["had"] += [mass]
            self.jets_leps["all"] += [mass]

            ntj = str(len(t.Jets)) + " - Jets"
            if len(ch_):
                if ntj not in self.n_jets_lep: self.n_jets_lep[ntj] = []
                self.n_jets_lep[ntj] += [mass]
            else:
                if ntj not in self.n_jets_had: self.n_jets_had[ntj] = []
                self.n_jets_had[ntj] += [mass]




