from AnalysisTopGNN.Templates import Selection 

class TopDecayModes(Selection):
    def __init__(self):
        Selection.__init__(self)
        self.CounterPDGID = {
                        "d"            : 0, "u"       : 0, "s"              : 0, "c"    : 0, 
                        "b"            : 0, "e"       : 0, "$\\nu_e$"       : 0, "$\mu$": 0, 
                        "$\\nu_{\mu}$" : 0, "$\\tau$" : 0, "$\\nu_{\\tau}$" : 0, "g"    : 0, 
                        "$\\gamma$"    : 0
        }
        self.TopCounter = 0
        self.TopMassTC = {"Had" : [], "Lep" : []}

    def Selection(self, event):
        if len(event.Tops) != 4:
                return False
        for t in event.Tops:
            if len(t.Children) == 0:
                return False
    
    def Strategy(self, event):
        PDGID = { 
                 1 : "d"        ,  2 : "u"             ,  3 : "s", 
                 4 : "c"        ,  5 : "b"             , 11 : "e", 
                12 : "$\\nu_e$" , 13 : "$\mu$"         , 14 : "$\\nu_{\mu}$", 
                15 : "$\\tau$"  , 16 : "$\\nu_{\\tau}$", 21 : "g", 
                22 : "$\\gamma$"
        }

        for t in event.Tops:
            self.TopCounter += 1
            for c in t.Children:
                pdg = PDGID[abs(c.pdgid)]
                self.CounterPDGID[pdg] += 1
            _leptons = [11, 12, 13, 14, 15, 16]
            mode = "Lep" if sum([i for i in t.Children if abs(i.pdgid) in _leptons]) else "Had"
            self.TopMassTC[mode] += [ sum(t.Children).Mass ]