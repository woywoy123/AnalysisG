from AnalysisG.Templates import SelectionTemplate

class TopMassJets(SelectionTemplate):
    
    def __init__(self):
        SelectionTemplate.__init__(self)
      
        self.TopMassTruthJet = {"Had" : [], "Lep" : []}
        self.TopMassJet = {"Had" : [], "Lep" : []}
        self.TopMassJetDetectorLep = {"Had" : [], "Lep" : []}

        self.NJets = {"Had" : [], "Lep" : []}
        self.NTruthJets = {"Had" : [], "Lep" : []}

        self.DeltaRJets = {"Had" : [], "Lep" : []}
  
    def Selection(self, event):
        return True 

    def Strategy(self, event):
        tops = event.Tops 
        lep = event.Electrons + event.Muons
        leptop = set([t for l in lep for t in l.Parent])
        for t in tops:
            fragtj, fragj, fragjl = [], [], []
            fragtj += t.TruthJets
            fragj += list(t.Jets)
            fragjl += list(t.Jets)
            if len(fragtj) == 0 or len(fragj) == 0: continue
            mode = "Lep" if t.LeptonicDecay else "Had"
            
            if t.LeptonicDecay: fragtj += [c for c in t.Children if c.is_nu or c.is_lep]
            if t.LeptonicDecay: fragj += [c for c in t.Children if c.is_nu or c.is_lep]
            
            self.TopMassTruthJet[mode] += [sum(fragtj).Mass/1000]
            self.TopMassJet[mode] += [sum(fragj).Mass/1000]
            
            self.NJets[mode] += [len(t.Jets)]
            self.NTruthJets[mode] += [len(t.TruthJets)]

            dr = []
            acc = []
            for p in fragj:
                for p_ in fragj:
                    if p == p_ or p_ in acc: continue
                    dr.append(p.DeltaR(p_))
                acc.append(p)
            if len(dr) == 0: dr.append(-1)
            self.DeltaRJets[mode].append(sum(dr)/len(dr))
 
            if t.LeptonicDecay and t not in leptop: continue
            fragjl += [l for l in lep if t in l.Parent] + [c for c in t.Children if c.is_nu]
            self.TopMassJetDetectorLep[mode] += [sum(fragjl).Mass/1000]

            
        



