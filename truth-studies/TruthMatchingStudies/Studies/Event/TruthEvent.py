from AnalysisTopGNN.Templates import Selection

class EventNTruthJetAndJets(Selection):

        def __init__(self):
                Selection.__init__(self)
                self.TruthJets = {}
                self.Jets = {}
                self.MET = []
                self.nLep = []
                
        def Strategy(self, event):
                lep = [11, 13, 15]
                ntjets = len(event.TruthJets)
                njets = len(event.Jets)

                if ntjets not in self.TruthJets:
                        self.TruthJets[ntjets] = 0
                if njets not in self.Jets:
                        self.Jets[njets] = 0
                self.TruthJets[ntjets] += 1
                self.Jets[njets] += 1
                self.MET.append(event.met)
                self.nLep.append(len([ i for i in event.TopChildren if abs(i.pdgid) in lep ]))
