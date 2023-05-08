from AnalysisG.Templates import SelectionTemplate

class ResonanceTruthJets(SelectionTemplate):
    
    def __init__(self):
        SelectionTemplate.__init__(self)

        self.ResMass = {"Lep-Had": [], "Lep-Lep" : [], "Had-Had" : []}


    def Selection(self, event):
        if len([i.FromRes for i in event.Tops if i.FromRes == 1]) != 2: return False
        return True

    def Strategy(self, event):
        for i in event.TruthJets:
            print([i])
