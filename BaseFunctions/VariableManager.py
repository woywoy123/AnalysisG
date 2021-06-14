from BaseFunctions.IO import FastReading
from BaseFunctions.Alerting import WarningAlert

class BranchVariable(WarningAlert):
    def __init__(self, FileDir, Tree, Branches):
        WarningAlert.__init__(self) 
        
        Branches = list(set(Branches))
        if "eventNumber" not in Branches:
            Branches.insert(0, "eventNumber")

        self.EventObjectMap = {}
        
        reader = FastReading(FileDir)
        reader.ReadBranchFromTree(Tree, Branches)
        reader.ConvertBranchesToArray()
        self.__Reader = reader.ArrayBranches[Tree]
        self.__Branches = Branches
        
        self.AssignBranchToVariable()

    def AssignBranchToVariable(self):
         
        for i in self.__Branches:
            self.__bI = i
            
            var = i.split("_")
            self.__Variable = var[len(var) -1]
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
            self.Mask = self.__Reader[self.__bI]
        
        if self.__Variable == "flavour":
            self.Flavour = self.__Reader[self.__bI]

        if self.__Variable == "charge":
            self.Charge = self.__Reader[self.__bI]

        if self.__Variable == "truthflav":
            self.TruthFlavour = self.__Reader[self.__bI]

        if self.__Variable == "extended":
            self.Extended = self.__Reader[self.__bI]

        if self.__Variable == "nCHad":
            self.nCHad = self.__Reader[self.__bI]

        if self.__Variable == "nBHad":
            self.nBHad = self.__Reader[self.__bI]

        if self.__Variable == "topoetcone20":
            self.topoetcone20 = self.__Reader[self.__bI]

        if self.__Variable == "ptvarcone20":
            self.ptvarcone20 = self.__Reader[self.bI]

        if self.__Variable == "CF":
            self.CF = self.__[self.bI]

        if self.__Variable == "d0sig":
            self.d0sig = self.__Reader[self.__bI]

        if self.__Variable == "sintheta":
            self.deltaz0sintheta = self.__Reader[self.__bI]

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
