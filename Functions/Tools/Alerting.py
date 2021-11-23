import uproot

class Notification():
    def __init__(self, Verbose = False):
        self.Verbose = Verbose
        self.__INFO = "INFO"
        self.__FAIL = "FAILURE"
        self.__WARNING = "WARNING"
        self.Caller = ""
        self.Alerted = []

    def Notify(self, Message):
        if self.Verbose and self.Caller == "":
            print(self.__INFO + Message)
        elif self.Verbose and self.Caller != "":
            print(self.Caller.upper()+"::"+self.__INFO+"::"+Message)

    def NothingGiven(self, Obj, Name):
        if len(Obj) == 0:
            print(self.Caller.upper()+"::"+self.__WARNING+"::NOTHING GIVEN +-> " + Name)


    def CheckObject(self, Object, Key):
        try:
            Object[Key]
            return True
        except uproot.exceptions.KeyInFileError:
            if Key not in self.Alerted:
                print(self.__WARNING + self.Caller + " :: Key -> " + Key + " NOT FOUND")
                self.Alerted.append(Key)
            return False
    
    def Warning(self, text):
        print(self.__WARNING + self.Caller + " :: " + text)

    def CheckAttribute(self, Obj, Attr):
        try: 
            getattr(Obj, Attr)
            return True
        except AttributeError:
            return False

class Debugging:

    def __init__(self, Threshold = 100):
        self.__Threshold = Threshold 
        self.__iter = 0
        self.Debug = False
    
    def TimeOut(self):
        if self.__Threshold <= self.__iter and self.__Threshold != -1:
            return True
        self.__iter +=1
        
    def ResetCounter(self):
        self.__iter = 0

    def DebugTruthDetectorMatch(self, All):
        
        truthjet_l = []
        jet_l = []
        child = []
        print(">>>>>>>========== " + self.CallLoop + " ==========")
        print("    --- All Particles ---  ")
        for i in All:
            if i.Type == "jet":
                print("Type: ", i.Type, " FL: ", i.truthflav, "-- FLE: ", i.truthflavExtended, "--- TruthParton: ", i.truthPartonLabel)
                jet_l.append(i)

            if i.Type == "mu" or i.Type == "el":
                print("Type: ", i.Type, " TruTyp: ", i.true_type, " TruOri: ", i.true_origin, " Prompt: ", i.true_isPrompt, " C: ", i.charge)

            if i.Type == "truthjet":
                print("Type: ", i.Type, " FL: ", i.flavour, " FLE: ", i.flavour_extended, " nCHad: ", i.nCHad, " nBHad: ", i.nBHad, " Index: ", i.Index)
                truthjet_l.append(i) 
                for j in i.Decay:
                    print("Type: ", j.Type, " FL: ", j.truthflav, " FLE: ", j.truthflavExtended, 
                          " TP: ", j.truthPartonLabel, " HS: ", j.isTrueHS, "Index: ", j.Index)

            if i.Type == "truth_top_child" or i.Type == "truth_top_initialState_child":
                print("Type: ", i.Type, " PDGID: ", i.pdgid, " Index: ", i.Index)
                child.append(i)
        
        print("")
        print("####### dR ########")
        captured = []
        for dR in self.dRMatrix:
            jet = dR[1][0]
            tjet = dR[1][1]
            dR = dR[1][2]
 
            if jet in captured:
                continue

            if self.MatchingRule(tjet, jet, dR):
                if "init" in tjet.Type:
                    tjet.Decay_init.append(jet)
                else:
                    tjet.Decay.append(jet)
                captured.append(jet) 
            
            if tjet.Type == "truthjet" and jet.Type == "jet":
                print("TJet Flavour: ", tjet.flavour, " Extended: ", tjet.flavour_extended, 
                        " nChad", tjet.nCHad, " nBhad", tjet.nBHad, " Index: ", tjet.Index,
                      "|| Jet Flavour: ", jet.truthflav, " Extended: ", jet.truthflavExtended, 
                      " Truth Parton: ", jet.truthPartonLabel, " HS: ", jet.isTrueHS, " dR: ", round(dR, 4))
            
            if "child" in tjet.Type and jet.Type == "truthjet":
                print("Child PDGID: ", tjet.pdgid, " Index: ", tjet.Index, 
                      "|| Jet Flavour: ", jet.flavour, " Extended: ", jet.flavour_extended, 
                      " nChad", jet.nCHad, " nBhad", jet.nBHad, " dR: ", round(dR, 4))

            if "child" in tjet.Type and (jet.Type == "el" or jet.Type == "mu"):
                print("Child PDGID: ", tjet.pdgid, " Index: ", tjet.Index, 
                      "|| Type: ", jet.Type, " TruTyp: ", jet.true_type, 
                      " TruOri: ", jet.true_origin, " Prompt: ", jet.true_isPrompt, " C: ", jet.charge , "dR: ", round(dR, 4))
        
        if len(truthjet_l) != 0 and len(jet_l) != 0:
            print("===== Result::")
            print("TJ: ", len(truthjet_l), "J: ", len(jet_l), "Captured: ", len(captured))
            
            for tjet in truthjet_l:
                print("(TRUTHJET-JET-Match) Decay Product number: ", len(tjet.Decay), "FL: ", tjet.flavour, " FLE: ", tjet.flavour_extended, " Index: ", tjet.Index)
                for jet in tjet.Decay:
                    print("---> TJet Flavour: ", tjet.flavour, " Extended: ", tjet.flavour_extended, 
                          " nChad", tjet.nCHad, " nBhad", tjet.nBHad, " || Jet Flavour: ", jet.truthflav, " Extended: ", jet.truthflavExtended, 
                          " Truth Parton: ", jet.truthPartonLabel, " HS: ", jet.isTrueHS, " dR: ", round(tjet.DeltaR(jet), 4))

        if len(truthjet_l) != 0 and len(child) != 0:
            print("===== Result::")
            print("TJ: ", len(truthjet_l), "Child: ", len(child), "Captured: ", len(captured))
            
            for c in child:
                if "init" in c.Type:
                    print("(TOP-CHILD-TRUTHJET-Match) Decay Product number: ", len(c.Decay_init), "PDGID: ", c.pdgid, " Index: ", c.Index)
                    it = c.Decay_init
                else:
                    print("(TOP-CHILD-TRUTHJET-Match) Decay Product number: ", len(c.Decay), "PDGID: ", c.pdgid, " Index: ", c.Index)
                    it = c.Decay
                
                for tjet in it:
                    if tjet.Type == "truthjet":
                        print("+--> TJet Flavour: ", tjet.flavour, " Extended: ", tjet.flavour_extended, 
                              " nChad", tjet.nCHad, " nBhad", tjet.nBHad, " dR: ", round(tjet.DeltaR(c), 4))
                        for j in tjet.Decay:
                            print("+> Jet Flavour: ", j.truthflav, " Extended: ", j.truthflavExtended, 
                                  " Truth Parton: ", j.truthPartonLabel, " HS: ", j.isTrueHS, " dR: ", round(j.DeltaR(tjet), 4))
                    else: 
                        print("+> Type: ", tjet.Type, " TruTyp: ", tjet.true_type, " TruOri: ", tjet.true_origin, 
                              " Prompt: ", tjet.true_isPrompt, " C: ", tjet.charge , "dR: ", round(tjet.DeltaR(c), 4))
                print("")
 
        print("")


