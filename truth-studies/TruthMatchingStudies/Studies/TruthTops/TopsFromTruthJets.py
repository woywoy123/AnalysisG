from AnalysisG.Templates import SelectionTemplate

class TopMassTruthJets(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)

        self.TopMass = {"Had" : [], "Lep" : []}
        self.TruthTopMass = {"Had" : [], "Lep" : []}

        self.TopMassNjets = {}
        self.TopMassMerged = {}

    def Selection(self, event):
        return True 

    def Strategy(self, event):
        tops = event.Tops    
        for t in tops:
            frag = []
            frag += t.TruthJets
            if len(t.TruthJets) == 0: continue
            mode = "Lep" if t.LeptonicDecay else "Had"
            if t.LeptonicDecay:
                frag += [c for c in t.Children if c.is_lep or c.is_nu]
            self.TopMass[mode] += [sum(frag).Mass / 1000]
            self.TruthTopMass[mode] += [t.Mass / 1000]
            
            ntj = mode + "-" + str(len(t.TruthJets))
            if ntj not in self.TopMassNjets: self.TopMassNjets[ntj] = []
            self.TopMassNjets[ntj] += [sum(frag).Mass / 1000] 
          
            merged = len(set([_t for tj in t.TruthJets for _t in tj.Tops]))
            if merged not in self.TopMassMerged: self.TopMassMerged[merged] = []
            self.TopMassMerged[merged] += [sum(frag).Mass / 1000]

class TopTruthJetsKinematics(SelectionTemplate):
    
    def __init__(self):
        SelectionTemplate.__init__(self)
        self.DeltaRTJ_ = {
                    "Spec-Energy" : [], "Res-Energy" : [], 
                    "Spec-PT" : [], "Res-PT" : [], 
                    "Spec-dR" : [], "Res-dR" : [], 
                    "Spec-bkg-dR" : [], "Res-bkg-dR" : [], 
                    "Spec-bkg-Energy" : [], "Res-bkg-Energy" : [], 
                    "Spec-bkg-PT" : [], "Res-bkg-PT" : [], 
        }
        self.TopTruthJet_parton = {
                    "Parton-dR" : [], 
                    "Parton-symbol" : [], 
                    "Parton-PT-Frac" : [], 
                    "Parton-Energy-Frac" : [], 
                    "Parton-eta" : [], 
                    "Parton-phi" : [], 
        }

        self.TopMass = {
                    "NoGluons" : [], 
                    "Nominal" : [],
        } 

        self.JetMassNTop = {}
   
    def Selection(self, event):
        return True  

    def Strategy(self, event):
        tops = event.Tops         
        for t in tops:
            col, gluonless = [], []
            mode = "Res" if t.FromRes == 1 else "Spec"
            for tj1 in t.TruthJets: 
                for tj2 in t.TruthJets: 
                    if tj2 in col or tj1 == tj2: continue
                    dr = tj1.DeltaR(tj2)
                    self.DeltaRTJ_[mode + "-dR"].append(dr)
                    self.DeltaRTJ_[mode + "-Energy"].append(t.e/1000)
                    self.DeltaRTJ_[mode + "-PT"].append(t.pt/1000)
                col.append(tj1)
                for tj in event.TruthJets:
                    if tj in t.TruthJets: continue 
                    dr = tj1.DeltaR(tj)
                    self.DeltaRTJ_[mode + "-bkg-dR"].append(dr)
                    self.DeltaRTJ_[mode + "-bkg-Energy"].append(t.e/1000) 
                    self.DeltaRTJ_[mode + "-bkg-PT"].append(t.pt/1000)
            
                for prt in tj1.Parton:
                    dr = tj1.DeltaR(prt)
                    sym = "None" if prt.symbol == "" else prt.symbol

                    self.TopTruthJet_parton["Parton-eta"].append(prt.eta)                   
                    self.TopTruthJet_parton["Parton-phi"].append(prt.phi)                   
                    self.TopTruthJet_parton["Parton-dR"].append(dr)
                    
                    self.TopTruthJet_parton["Parton-symbol"].append(sym)
                    self.TopTruthJet_parton["Parton-Energy-Frac"].append(prt.e / tj1.e)

                ntops = len(tj1.Tops)
                if ntops not in self.JetMassNTop: self.JetMassNTop[ntops] = []
                self.JetMassNTop[ntops].append(tj1.Mass / 1000)

                if len([prt for prt in tj1.Parton if prt.symbol != "g"]) == 0: continue
                gluonless.append(tj1)

            if t.LeptonicDecay: continue
            if len(gluonless) == 0: continue
            self.TopMass["NoGluons"].append(sum(gluonless).Mass/1000)
            self.TopMass["Nominal"].append(sum(t.TruthJets).Mass/1000)

