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
        if self.__Variable == "eventNumber":
            self.EventNumber = self.__Reader[self.__bI]
            self.Type = self.__Variable


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

        if self.__Variable == "met":
            self.MET = self.__Reader[self.__bI]

        if self.__Variable == "charge":
            self.Charge = self.__Reader[self.__bI]



