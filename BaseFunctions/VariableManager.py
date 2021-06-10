from BaseFunctions.IO import FastReading
from BaseFunctions.Alerting import WarningAlert

class BranchVariable(WarningAlert):
    def __init__(self, FileDir, Tree, Branches):
        WarningAlert.__init__(self) 

        reader = FastReading(FileDir)
        reader.ReadBranchFromTree(Tree, Branches)
        reader.ConvertBranchesToArray()
        self.__Reader = reader.ArrayBranches[Tree]
        self.__Branches = Branches
            
        self.EventObjectMap = {}
        
        self.AssignBranchToVariable()

    def AssignBranchToVariable(self):
         
        for i in self.__Branches:
            self.__bI = i
            
            var = i.split("_")
            self.__Variable = var[len(var) -1]
            self.Type = "_".join(var[0:len(var) -1])
           
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





