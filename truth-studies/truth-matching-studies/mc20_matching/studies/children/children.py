from AnalysisG.Templates import SelectionTemplate

class ChildrenKinematics(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)

        self.dr_children_top = {}
        self.dr_children_cluster = {"Mutual" : [], "non-Mutual" : []}

    def Selection(self, event): return True

    def Strategy(self, event):
        top = event.Tops
        for t in top:
            mode = [c for c in t.Children if c.is_nu]
            if len(mode): mode = "lep"
            else: mode = "had"

            if mode in self.dr_children_top: pass
            else: self.dr_children_top[mode] = {"all" : []}

            for c in t.Children:
                self.dr_children_top[mode]["all"] += [t.DeltaR(c)]
                if c.symbol in self.dr_children_top[mode]: pass
                else: self.dr_children_top[mode][c.symbol] = []
                self.dr_children_top[mode][c.symbol] += [t.DeltaR(c)]


        scanned = []
        for c1 in event.TruthChildren:
            for c2 in event.TruthChildren:
                if c1 == c2: continue
                if c2 in scanned: continue
                mut = "Mutual"
                if c1.Parent[0] != c2.Parent[0]: mut = "non-Mutual"
                self.dr_children_cluster[mut] += [c1.DeltaR(c2)]
                scanned += [c1]
