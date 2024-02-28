from AnalysisG.Templates import SelectionTemplate

class MergedTopsJets(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)
        self.PartonPT = {}
        self.PartonEnergy = {}
        self.PartonDr = {}
        self.ChildPartonPT = {}
        self.ChildPartonEnergy = {}
        self.ChildPartonDr = {}
        self.dRChildPartonJetAxis = {}
        self.NumberOfConstituentsInJet = {}
        self.TopsJets = {}
        self.TopsJetsMerged = {}
        self.TopsJetsNoPartons = {}
        self.TopsJetsCut = {0.95 : [], 0.9 : [], 0.8 : [], 0.7 : []}

    def Selection(self, event):
        if len(event.Tops) != 4: return False
        if len([i for i in event.Tops if i.LeptonicDecay]) > 2: return False
        return True 

    def Strategy(self, event):
        jets = event.Jets
        for tj in jets:
            if len(tj.Tops) < 2: continue
            tmp = {}
            
            for ptrn in tj.Parton:
                if ptrn.symbol not in self.PartonDr:
                    self.PartonPT[ptrn.symbol] = []
                    self.PartonEnergy[ptrn.symbol] = []
                    self.PartonDr[ptrn.symbol] = []
                if ptrn.symbol not in tmp: tmp[ptrn.symbol] = 0
                tmp[ptrn.symbol] += 1

                self.PartonPT[ptrn.symbol].append(ptrn.pt/1000)
                self.PartonEnergy[ptrn.symbol].append(ptrn.e/1000)
                self.PartonDr[ptrn.symbol].append(ptrn.DeltaR(tj))

                for c in ptrn.Parent:
                    if ptrn.symbol not in self.ChildPartonPT:
                        self.ChildPartonPT[ptrn.symbol] = []
                        self.ChildPartonEnergy[ptrn.symbol] = [] 
                        self.ChildPartonDr[ptrn.symbol] = []      
                        self.dRChildPartonJetAxis[ptrn.symbol] = []           

                    self.ChildPartonPT[ptrn.symbol].append(c.pt/1000)
                    self.ChildPartonEnergy[ptrn.symbol].append(c.e/1000)
                    self.ChildPartonDr[ptrn.symbol].append(ptrn.DeltaR(c))               
                    self.dRChildPartonJetAxis[ptrn.symbol].append(tj.DeltaR(c))

            for key in tmp: 
                if key not in self.NumberOfConstituentsInJet: self.NumberOfConstituentsInJet[key] = []
                self.NumberOfConstituentsInJet[key].append(tmp[key])

        for t in event.Tops:
            if t.LeptonicDecay: continue 
            if len(t.Jets) == 0: continue
            nmerged = len(set([x for tj in t.Jets for x in tj.Tops]))
            if nmerged not in self.TopsJets: 
                self.TopsJets[nmerged] = []
                self.TopsJetsNoPartons[nmerged] = []
            self.TopsJets[nmerged].append(sum(t.Jets).Mass / 1000)
            tjets = []
            tj_cuts = {0.95 : [], 0.9 : [], 0.8 : [], 0.7 : []}
            for tj in t.Jets: 
                prt_energy_all = []
                prt_this_top = [] 
                ntops = len(tj.Tops)
                if len(tj.Parton) == 0: continue
                for prt in tj.Parton:
                    prt_energy_all.append(prt.e/1000) 
                    prt_this_top += [prt.e / 1000] if t == prt.Parent[0].Parent[0] else []
                frac = sum(prt_this_top) / sum(prt_energy_all)
                if ntops not in self.TopsJetsMerged: self.TopsJetsMerged[ntops] = []
                self.TopsJetsMerged[ntops].append(frac)
                tjets.append(tj) 
               
                for cut in tj_cuts:
                    if frac < cut: continue
                    if nmerged == 1: continue
                    tj_cuts[cut].append(tj)
            if len(tjets) == 0: continue
            self.TopsJetsNoPartons[nmerged].append(sum(tjets).Mass/1000)
            for cut in self.TopsJetsCut:
                if len(tj_cuts[cut]) == 0: continue 
                self.TopsJetsCut[cut].append(sum(tj_cuts[cut]).Mass/1000)
