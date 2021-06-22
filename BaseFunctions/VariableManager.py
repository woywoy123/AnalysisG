from BaseFunctions.IO import FastReading
from BaseFunctions.Alerting import WarningAlert

def Init():
    global ConvertedBranches
    global FilesConverted 
    FilesConverted = []
    ConvertedBranches= {}

class BranchVariable(WarningAlert):
    def __init__(self, FileDir, Tree, Branches):
        WarningAlert.__init__(self) 
        
        Branches = list(set(Branches))
        if "eventNumber" not in Branches:
            Branches.insert(0, "eventNumber")
        
        if FileDir not in FilesConverted:
            FilesConverted.append(FileDir)
            ConvertedBranches.clear()

        self.__Reader = {}
        R_B = []
        for i in Branches:
            s = str(Tree + "/" + i)
            
            try:
                self.__Reader[i] = ConvertedBranches[s]
            except KeyError:
                R_B.append(i)

        if len(R_B) != 0:
            reader = FastReading(FileDir)
            reader.ReadBranchFromTree(Tree, R_B)
            reader.ConvertBranchesToArray()
            
            for i in R_B:
                ConvertedBranches[str(Tree + "/" + i)] = reader.ArrayBranches[Tree][i]
                self.__Reader[i] = reader.ArrayBranches[Tree][i]


        self.__Branches = []
        for i in self.__Reader:
            self.__Branches.append(i)
        
        self.EventObjectMap = {}
        self.AssignBranchToVariable()

    def AssignBranchToVariable(self):
         
        for i in self.__Branches:
            self.__bI = i
            
            var = i.split("_")
            self.__Variable = var[len(var) -1]

            if "truthjet_" in self.__bI:
                self.Type = "TruthJet"
            elif "el_" in self.__bI:
                self.Type = "Electron"
            elif "mu_" in self.__bI:
                self.Type = "Muon"
            elif "jet_" in self.__bI:
                self.Type = "Jet"
            elif "top_" in self.__bI:
                self.Type = "Top"
            elif "rcjet" in self.__bI:
                self.Type = "RCJet"
            else:
                self.Type = var[0]
           
            self.EventObjectProperties()
            self.DetectorObjectProperties()
             
            self.MixingTypes(self.Type)

    def EventObjectProperties(self):
        def CreateEventMap():
            for i in self.EventNumber:
                self.EventObjectMap[i] = {}

        def FillEventMap(EventObject):
            for i in range(len(self.EventNumber)):
                self.EventObjectMap[self.EventNumber[i]][self.__bI] = EventObject[i]

        
        if self.__bI == "eventNumber":
            self.EventNumber = self.__Reader[self.__bI]
            CreateEventMap()

        if self.__bI == "met_met":
            self.MET = self.__Reader[self.__bI]
            FillEventMap(self.MET)

        if self.__bI == "met_phi":
            self.MET_Phi = self.__Reader[self.__bI]
            FillEventMap(self.MET_Phi)

        if self.__bI == "ttbar_type_truthjets":
            self.ttbar_truthjets = self.__Reader[self.__bI]
            FillEventMap(self.ttbar_truthjets)


    def DetectorObjectProperties(self):
        if self.__Variable == "phi":
            self.Phi = self.__Reader[self.__bI]

        if self.__Variable == "eta":
            self.Eta = self.__Reader[self.__bI]
       
        if self.__Variable == "pt":
            self.Pt = self.__Reader[self.__bI]
        
        if self.__Variable == "e":
            self.E = self.__Reader[self.__bI]
        
        if self.__Variable == "pdgid":
            self.PDGID = self.__Reader[self.__bI]

        if self.__Variable == "FromRes":
            self.IsSignal = self.__Reader[self.__bI]
        
        if self.__Variable == "flavour":
            self.Flavour = self.__Reader[self.__bI]

        if self.__Variable == "charge":
            self.Charge = self.__Reader[self.__bI]

        if self.__Variable == "truthflav":
            self.truthflav = self.__Reader[self.__bI]

        if self.__Variable == "extended":
            self.Extended = self.__Reader[self.__bI]

        if self.__Variable == "nCHad":
            self.nChad = self.__Reader[self.__bI]

        if self.__Variable == "nBHad":
            self.nBhad = self.__Reader[self.__bI]

        if self.__Variable == "topoetcone20":
            self.topoetcone20 = self.__Reader[self.__bI]

        if self.__Variable == "ptvarcone20":
            self.ptvarcone20 = self.__Reader[self.__bI]

        if self.__Variable == "ptvarcone30":
            self.ptvarcone30 = self.__Reader[self.__bI]

        if self.__Variable == "CF":
            self.CF = self.__Reader[self.__bI]

        if self.__Variable == "d0sig":
            self.d0sig = self.__Reader[self.__bI]

        if self.__Variable == "sintheta":
            self.delta_z0_sintheta = self.__Reader[self.__bI]

        if self.__Variable == "type":
            self.true_type = self.__Reader[self.__bI]

        if self.__Variable == "origin":
            self.true_origin = self.__Reader[self.__bI]
        
        if self.__Variable == "firstEgMotherTruthType":
            self.true_firstEgMotherTruthType = self.__Reader[self.__bI]
        
        if self.__Variable == "firstEgMotherTruthOrigin":
            self.true_firstEgMotherTruthOrigin = self.__Reader[self.__bI]
       
        if self.__Variable == "firstEgMotherPdgId":
            self.true_firstEgMotherPdgId = self.__Reader[self.__bI]

        if self.__Variable == "IFFclass":
            self.true_IFFclass = self.__Reader[self.__bI]

        if self.__Variable == "isPrompt":
            self.true_isPrompt = self.__Reader[self.__bI]
                
        if self.__Variable == "isChargeFl":
            self.true_isChargeFl = self.__Reader[self.__bI]

        if self.__Variable == "jvt":
            self.jvt = self.__Reader[self.__bI]

        if self.__Variable == "truthPartonLabel":
            self.truthPartonLabel = self.__Reader[self.__bI]

        if self.__Variable == "isTrueHS":
            self.isTrueHS = self.__Reader[self.__bI]

        if "DL1r" in self.__bI:

            if self.__Variable == "60":
                self.DL1r_60 = self.__Reader[self.__bI]

            if self.__Variable == "70":
                self.DL1r_70 = self.__Reader[self.__bI]

            if self.__Variable == "77":
                self.DL1r_77 = self.__Reader[self.__bI]

            if self.__Variable == "85":
                self.DL1r_85 = self.__Reader[self.__bI]

            if self.__Variable == "DL1r":
                self.DL1r = self.__Reader[self.__bI]

            if self.__Variable == "pb":
                self.DL1r_pb = self.__Reader[self.__bI]

            if self.__Variable == "pc":
                self.DL1r_pc = self.__Reader[self.__bI]

            if self.__Variable == "pu":
                self.DL1r_pu = self.__Reader[self.__bI]


        elif "DL1" in self.__bI:

            if self.__Variable == "60":
                self.DL1_60 = self.__Reader[self.__bI]

            if self.__Variable == "70":
                self.DL1_70 = self.__Reader[self.__bI]

            if self.__Variable == "77":
                self.DL1_77 = self.__Reader[self.__bI]

            if self.__Variable == "85":
                self.DL1_85 = self.__Reader[self.__bI]

            if self.__Variable == "DL1":
                self.DL1 = self.__Reader[self.__bI]

            if self.__Variable == "pb":
                self.DL1_pb = self.__Reader[self.__bI]

            if self.__Variable == "pc":
                self.DL1_pc = self.__Reader[self.__bI]

            if self.__Variable == "pu":
                self.DL1_pu = self.__Reader[self.__bI]

def VariableObjectProxy(self, P, i, j = -1):
    def AssignVariableToObject(self, Object, Variable, i_i, i_j = -1):
        try: 
            x = getattr(self, Variable)
            if isinstance(x, str):
                return
            if i_j == -1:
                setattr(Object, Variable, x[i_i])
            else:
                setattr(Object, Variable, x[i_i][i_j])
        except AttributeError:
            pass


    if self.Type == "Jet" or self.Type == "TruthJet" or self.Type == "RCJet":
        AssignVariableToObject(self, P, "Flavour", i, j)
        AssignVariableToObject(self, P, "nChad", i, j)
        AssignVariableToObject(self, P, "nBhad", i, j)
        AssignVariableToObject(self, P, "truthflav", i, j)
        AssignVariableToObject(self, P, "truthPartonLabel", i, j)
        AssignVariableToObject(self, P, "isTrueHS", i, j)
        AssignVariableToObject(self, P, "DL1r", i, j)
        AssignVariableToObject(self, P, "DL1r_60", i, j)
        AssignVariableToObject(self, P, "DL1r_70", i, j)
        AssignVariableToObject(self, P, "DL1r_77", i, j)
        AssignVariableToObject(self, P, "DL1r_85", i, j)
        AssignVariableToObject(self, P, "DL1r_pb", i, j)
        AssignVariableToObject(self, P, "DL1r_pc", i, j)
        AssignVariableToObject(self, P, "DL1r_pu", i, j)
        AssignVariableToObject(self, P, "DL1", i, j)
        AssignVariableToObject(self, P, "DL1_60", i, j)
        AssignVariableToObject(self, P, "DL1_70", i, j)
        AssignVariableToObject(self, P, "DL1_77", i, j)
        AssignVariableToObject(self, P, "DL1_85", i, j)
        AssignVariableToObject(self, P, "DL1_pb", i, j)
        AssignVariableToObject(self, P, "DL1_pc", i, j)
        AssignVariableToObject(self, P, "DL1_pu", i, j)
        AssignVariableToObject(self, P, "jvt", i, j)


    
    elif self.Type == "Muon" or self.Type == "Electron":
        AssignVariableToObject(self, P, "Charge", i, j)
        AssignVariableToObject(self, P, "PDGID", i, j)
        AssignVariableToObject(self, P, "topoetcone20", i, j)
        AssignVariableToObject(self, P, "ptvarcone20", i, j)
        AssignVariableToObject(self, P, "ptvarcone30", i, j)
        AssignVariableToObject(self, P, "CF", i, j)
        AssignVariableToObject(self, P, "d0sig", i, j)
        AssignVariableToObject(self, P, "delta_z0_sintheta", i, j)
        AssignVariableToObject(self, P, "true_type", i, j)
        AssignVariableToObject(self, P, "true_origin", i, j)
        AssignVariableToObject(self, P, "true_firstEgMotherTruthType", i, j)
        AssignVariableToObject(self, P, "true_firstEgMotherTruthOrigin", i, j)
        AssignVariableToObject(self, P, "true_firstEgMotherPdgId", i, j)
        AssignVariableToObject(self, P, "true_IFFclass", i, j)
        AssignVariableToObject(self, P, "true_isPrompt", i, j)
        AssignVariableToObject(self, P, "true_isChargeFl", i, j)

    else: 
        AssignVariableToObject(self, P, "Charge", i, j)
        AssignVariableToObject(self, P, "PDGID", i, j)
        AssignVariableToObject(self, P, "IsSignal", i, j)

