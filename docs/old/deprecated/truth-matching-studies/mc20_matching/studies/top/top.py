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


    def Strategy(self, event):
        tops = event.Tops
        self.truth_top += [t.Mass/1000 for t in tops]

        for t in tops:
            ch = t.Children
            if not len(ch): continue

            ch_ = [c for c in ch if c.is_lep]
            self.truth_children["all"] += [sum(ch).Mass/1000]

            # Children
            mode = "had"
            if len(ch_): mode = "lep"
            self.truth_children[mode] += [sum(ch).Mass/1000]

            # Truth Jets with Truth Children (leptonic decay)
            tru_dic = {t : {}}
            for tj in event.PhysicsTruth:
                for ti in tj.Parent:
                    try: tru_dic[ti][tj] = None
                    except KeyError: continue

            phys_tru = list(set(list(tru_dic[t])))
            ntj = str(len(phys_tru)) + " - Truth Jets"
            phys_tru = phys_tru + [c for c in ch if c.is_nu]
            if len(phys_tru):
                top_M = sum(phys_tru).Mass/1000
                self.truth_jets["all"] += [top_M]
                self.truth_jets[mode] += [top_M]
                if len(ch_):
                    if ntj not in self.n_truth_jets_lep: self.n_truth_jets_lep[ntj] = []
                    self.n_truth_jets_lep[ntj] += [top_M]
                else:
                    if ntj not in self.n_truth_jets_had: self.n_truth_jets_had[ntj] = []
                    self.n_truth_jets_had[ntj] += [top_M]

            # Jets with Truth Children (leptonic decay)
            jet_dic = {t : []}
            for jet in event.Jets:
                if t not in jet.Parent: continue
                jet_dic[t] += [jet]

            phys_det = list(set(list(jet_dic[t])))
            nj = str(len(phys_det)) + " - Jets"
            phys_det = phys_det + [c for c in ch if c.is_lep or c.is_nu]
            if len(phys_det):
                top_M = sum(phys_det).Mass/1000
                self.jets_truth_leps["all"] += [top_M]
                self.jets_truth_leps[mode] += [top_M]
                if len(ch_):
                    if nj not in self.n_jets_lep: self.n_jets_lep[nj] = []
                    self.n_jets_lep[nj] += [top_M]
                else:
                    if nj not in self.n_jets_had: self.n_jets_had[nj] = []
                    self.n_jets_had[nj] += [top_M]



            # Detector only objects (except neutrinos)
            jet_dic = {t : {}}
            for jet in event.PhysicsDetector:
                for ti in jet.Parent:
                    try: jet_dic[ti][jet] = None
                    except KeyError: continue

            phys_det = list(jet_dic[t]) + [c for c in ch if c.is_nu]
            phys_det = list(set(phys_det))
            if len(phys_det):
                top_M = sum(phys_det).Mass/1000
                self.jets_leps["all"] += [top_M]
                self.jets_leps[mode] += [top_M]






