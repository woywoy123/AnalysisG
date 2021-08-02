import uproot

class Notification():
    def __init__(self, Verbose = False):
        self.Verbose = Verbose
        self.__INFO = "INFO::"
        self.__FAIL = "FAILURE::"
        self.__WARNING = "WARNING::"
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
    
    def TimeOut(self):
        if self.__Threshold == self.__iter:
            return True
        self.__iter +=1
        
    def ResetCounter(self):
        self.__iter = 0

    def DebugTruthDetectorMatch(self, All):

        print(">>>>>>>========== New Event ==========")
        print("    --- All Particles ---  ")
        for i in All:
            if i.Type == "jet":
                print("Type: ", i.Type, " FL: ", i.truthflav, "-- FLE: ", i.truthflavExtended, "--- TruthParton: ", i.truthPartonLabel)

            if i.Type == "mu" or i.Type == "el":
                print("Type: ", i.Type, " TruTyp: ", i.true_type, " TruOri: ", i.true_origin, " Prompt: ", i.true_isPrompt, " C: ", i.charge)

            if i.Type == "truthjet":
                print("Type: ", i.Type, " FL: ", i.flavour, " FLE: ", i.flavour_extended, " nCHad: ", i.nCHad, " nBHad: ", i.nBHad)

            if i.Type == "truth_top_child" or i.Type == "truth_top_initialState_child":
                print("Type: ", i.Type, " PDGID: ", i.pdgid, " Index: ", i.Index)
        
        print("")
        print("####### dR ########")
        captured = []
        for dR in self.dRMatrix:
            jet = dR[1][0]
            tjet = dR[1][1]
            dR = dR[0]
 
            if self.MatchingRule(tjet, jet, dR):
                tjet.Decay.append(jet)
                captured.append(jet) 
            
            if jet in captured:
                continue
            

            if tjet.Type == "truthjet" and jet.Type == "jet":
                print("TJet Flavour: ", tjet.flavour, " Extended: ", tjet.flavour_extended, 
                      " nChad", tjet.nCHad, " nBhad", tjet.nBHad, " Parent PDGID: ", tjet.ParentPDGID,
                      "Jet Flavour: ", jet.truthflav, " Extended: ", jet.truthflavExtended, 
                      " Truth Parton: ", jet.truthPartonLabel, " HS: ", jet.isTrueHS, " dR: ", round(dR, 4))
            
            if "child" in tjet.Type and jet.Type == "truthjet":
                print("Child PDGID: ", tjet.pdgid, " Index: ", tjet.Index, 
                      "Jet Flavour: ", jet.flavour, " Extended: ", jet.flavour_extended, 
                      " nChad", jet.nCHad, " nBhad", jet.nBHad, " dR: ", round(dR, 4))

            if "child" in tjet.Type and (jet.Type == "el" or jet.Type == "mu"):
                print("Child PDGID: ", tjet.pdgid, " Index: ", tjet.Index, 
                      "Type: ", jet.Type, " TruTyp: ", jet.true_type, 
                      " TruOri: ", jet.true_origin, " Prompt: ", jet.true_isPrompt, " C: ", jet.charge , "dR: ", round(dR, 4))


        print("")


