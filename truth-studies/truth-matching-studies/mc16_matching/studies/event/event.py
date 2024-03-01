from AnalysisG.Template import SelectionTemplate

class TruthEvent(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)

        self.met_data = {"truth-children" : [], "delta" : [], "num_neutrino" : [], "met" : []}


    def Selection(self, event): return True

    def Strategy(self, event):
        met, phi = event.met/1000, event.phi
        nus = [i for i in event.TopChildren if i.is_nu]
        self.met_data["met"] += [met]
        self.met_data["num_neutrino"] += [len(nus)]
        self.met_data["truth-children"] += [sum(nus).pt/1000]
        self.met_data["delta"] += [abs(sum(nus).pt/1000 - met/1000)]
