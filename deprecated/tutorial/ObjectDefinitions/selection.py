from AnalysisG.Templates import SelectionTemplate

class ExampleSelection(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)
        self.__params__ = {"setting" : None}
        self.regions = {}
        self.nbjets = {}
        self.test = []

    def Selection(self, event):
        lep = event.Muons + event.Electrons
        jets = event.Jets

        sign = {}
        for i in lep:
            c = int(i.charge)
            if c not in sign: sign[c] = 0
            sign[c] += 1

        if -1 in sign and 1 in sign: return False
        if -1 in sign: return True
        if 1 in sign: return True

        nbs = [j for j in jets if j.dl1]
        if len(nbs) < 4: return False
        else: return True

    def Strategy(self, event):
        sign = {}
        lep = event.Muons + event.Electrons
        for i in lep:
            c = int(i.charge)
            if c not in sign: sign[c] = 0
            sign[c] += 1

        for i in sign:
            if i not in self.regions: self.regions[i] = 0
            self.regions[i] += sign[i]

        jets = event.Jets
        nbs = len([j for j in jets if j.dl1])
        if nbs not in self.nbjets: self.nbjets[nbs] = 0
        self.nbjets[nbs] += 1
        self.test.append(self.__params__["setting"])
