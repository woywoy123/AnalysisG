from BaseFunctions.IO import FastReading

class BranchVariable:
    def __init__(self, FileDir, Tree, Branches):
        self.Phi = "x"
        self.Eta = ""
        self.Pt = ""
        self.E = ""
        self.PDGID = ""
        self.Flavour = ""
        self.EventNumber = ""
        self.RunNumber = ""
        self.Charge = ""
        self.MET = ""
        self.Mask = ""
        self.__Branches = Branches
        self.ObjectType = ""


        assert(Tree, str)
        reader = FastReading(FileDir)
        reader.ReadBranchFromTree(Tree, Branches)
        reader.ConvertBranchesToArray()
        self.__Reader = reader.ArrayBranches[Tree]
            
        self.AssignBranchToVariable()

    def AssignBranchToVariable(self):
        
        for i in self.__Branches:
            self.__bI = i
            if i.count("_") == 0:
                self.EventObjectProperties()

            if i.count("_") == 1:
                self.__Variable = i.split("_")[1]
                self.Type = i.split("_")[0]
                self.DetectorObjectProperties()
        print(self.Phi) 
    
    def EventObjectProperties(self):
        if self.__Variable == "eventNumber":
            self.EventNumber = self.__Reader[self.__bI]

        if self.__Variable == "flavour":
            self.Flavour = self.__Reader[self.__bI]

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




