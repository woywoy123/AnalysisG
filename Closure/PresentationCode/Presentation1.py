from Closure.PresentationCode.GenericFunctions import *
from Functions.Plotting.TemplateHistograms import *
from Functions.IO.IO import UnpickleObject
from Functions.Event.DataLoader import GenerateDataLoader

Out = "PresentationPlots/Presentation1/Plots/"
def TopMassAnalysis(ev, Dir):
    print("Filename -> " + Dir + "/" + ev)
    eg = UnpickleObject(ev, Dir)

    Names = ["TruthTopMass", "Signal", "Spectator"]
    Backup = {i : [] for i in Names}
    for i in eg.Events:
        event = eg.Events[i]["nominal"]

        for t in event.TruthTops:
            t.CalculateMass()
            Backup["TruthTopMass"].append(t.Mass_GeV)
        
        for t in event.TopPostFSR:
            t.CalculateMassFromChildren()
            if len(t.Decay_init) == 0:
                continue
            
            if t.FromRes == 1:
                Backup["Signal"].append(t.Mass_init_GeV)
            else:
                Backup["Spectator"].append(t.Mass_init_GeV)
    BackupData("/".join(Out.split("/")[:-1]) + "/TopMassAnalysisBackup", DB = Backup, Name = ev)
  

def EdgeAnalysisOfChildren(ev, Dir):
    print("Filename -> " + Dir + "/" + ev)
    eg = UnpickleObject(ev, Dir)

    Names_m = ["SameParentMass", "DifferentParentMass"]
    Names_phi = ["SameParentPhi", "DifferentParentPhi"]
    Names_r = ["SameParentR", "DifferentParentR"]
    Names_E = ["SameParentE", "DifferentParentE"]
    Names_eta = ["SameParentEta", "DifferentParentEta"]
    Names_pt = ["SameParentPT", "DifferentParentPT"]   

    Backup = {}
    Backup |= {i : [] for i in Names_m}
    Backup |= {i : [] for i in Names_phi}
    Backup |= {i : [] for i in Names_r}
    Backup |= {i : [] for i in Names_E}
    Backup |= {i : [] for i in Names_eta}
    Backup |= {i : [] for i in Names_pt}

    for i in eg.Events:
        event = eg.Events[i]["nominal"]
        
        list1 = event.TopPostFSRChildren.copy()
        list2 = event.TopPostFSRChildren.copy()
        
        for c1 in list1:
            for c2 in list2:
                if c1 == c2:
                    continue
                m = Mass([c1, c2])
                phi = abs(c1.phi - c2.phi)
                r = c1.DeltaR(c2)
                e = abs(c1.e - c2.e)
                eta = abs(c1.eta - c2.eta)
                pt = abs(c1.pt - c2.pt)
                if c1.Index == c2.Index:
                    Backup["SameParentMass"].append(m)
                    Backup["SameParentPhi"].append(phi)
                    Backup["SameParentR"].append(r)
                    Backup["SameParentE"].append(e/1000)
                    Backup["SameParentEta"].append(eta)
                    Backup["SameParentPT"].append(pt/1000)
                else:
                    Backup["DifferentParentMass"].append(m)
                    Backup["DifferentParentPhi"].append(phi)
                    Backup["DifferentParentR"].append(r)
                    Backup["DifferentParentE"].append(e/1000)
                    Backup["DifferentParentEta"].append(eta)
                    Backup["DifferentParentPT"].append(pt/1000)
            list1.remove(c1)
    BackupData("/".join(Out.split("/")[:-1]) + "/EdgeAnalysisOfChildrenBackup", DB = Backup, Name = ev)

def CreatePlots(FileDir, CreateCache):
    CreateCache = False
    ev = CreateWorkspace("Presentation1", FileDir, CreateCache, -1)
    DL = GenerateDataLoader()
    
    for i in ev:
        Dir = "/".join(i.split("/")[:-1])
        Name = i.split("/")[-1]
        imprt = UnpickleObject(Name, Dir)
        DL.AddSample(imprt, "nominal", "TruthTopChildren", SelfLoop = True, FullyConnect = True)

        #TopMassAnalysis(Name, Dir)
        #EdgeAnalysisOfChildren(Name, Dir)

    
    #TopMassAnalysisPlots("Presentation1")
    #EdgeAnalysisOfChildrenPlots("Presentation1")
    return True






def TopMassAnalysisPlots(Dir):

    Backup = BackupData("/".join(Out.split("/")[:-1]) + "/TopMassAnalysisBackup", restore = True)

    H = TH1F(xData = Backup["TruthTopMass"], xBins = 250,
            Title = "Invariant Mass of Monte Carlo Truth Top", xTitle = "Mass (GeV)", yTitle = "Entries",
            OutputDirectory = Out + "TopMass", Filename = "TopMassTruth", 
            Style = "ATLAS")
    H.SaveFigure()

    H = TH1F(xData = Backup["Spectator"], xBins = 250, xMin = 165, 
            Title = "Reconstructed Monte Carlo Spectator \n Truth Top from Children (Pre-FSR)", xTitle = "Mass (GeV)", yTitle = "Entries",
            OutputDirectory = Out + "TopMass", Filename = "RecoSpectatorTop", 
            Style = "ATLAS")
    H.SaveFigure()


    H = TH1F(xData = Backup["Signal"], xBins = 250, xMin = 165, 
            Title = "Reconstructed Monte Carlo Signal \n Truth Top from Children (Pre-FSR)", xTitle = "Mass (GeV)", yTitle = "Entries",
            OutputDirectory = Out + "TopMass", Filename = "RecoSignalTop", 
            Style = "ATLAS")
    H.SaveFigure()

def EdgeAnalysisOfChildrenPlots(Dir):
    Backup = BackupData("/".join(Out.split("/")[:-1]) + "/EdgeAnalysisOfChildrenBackup", restore = True)
   
    #// ------- Constants -------- //#
    def Template(Title, xTitle, xMax, var, Filename):
        Out_D = Out + "EdgeFeatures/" 
        Params = {"yTitle" : "Entries", "xBins" : 500, "xMin" : 0, 
                "xTitle" : xTitle,  "xMax" : xMax, "xData" : None, 
                "Filename" : Filename, "OutputDirectory" : Out_D, 
                "Title" : Title, "Style" : "ATLAS", "Alpha" : 0.5}
         
        Params["Title"] = "Same Parent"
        Params["xData"] = Backup["SameParent" + var]
        H1 = TH1F(**Params)
        Params["Title"] = "Different Parent"   
        Params["xData"] = Backup["DifferentParent" + var]
        H2 = TH1F(**Params)
        Params["Title"] = Title
        HC = CombineTH1F(**Params)
        HC.Histograms = [H1, H2]
        HC.SaveFigure() 


    Template("Invariant Mass of Adjacent Nodes \n Sharing (un)common Top Parents", "Mass (GeV)", 1000, "Mass", "EdgeMass")
    Template("$ \Delta \phi $ of Adjacent Nodes Sharing (un)common Top Parents", "$ \Delta \phi $", 6, "Phi", "EdgeDeltaPhi")
    Template("$ \Delta R $ of Adjacent Nodes Sharing (un)common Top Parents", "$ \Delta R $", 6, "R", "EdgeDeltaR")
    Template("$ \Delta E $ of Adjacent Nodes Sharing (un)common Top Parents", "Energy (GeV)", 1000, "E", "EdgeDeltaE")
    
    Template("$ \Delta \eta $ of Adjacent Nodes Sharing (un)common Top Parents", "$ \Delta \eta $", 3, "Eta", "EdgeDeltaEta")
    Template("$ \Delta p_T $ of Adjacent Nodes Sharing (un)common Top Parents", "Transverse Momenta $ p_T $ (GeV)", 1000, "PT", "EdgeDeltaPT")







