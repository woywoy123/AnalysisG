from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event
from AnalysisTopGNN.IO import PickleObject, UnpickleObject
from AnalysisTopGNN.Plotting import TH1F, CombineTH1F
#import numpy as np
from itertools import combinations

PDGID = { 1 : "d"        ,  2 : "u"             ,  3 : "s", 
          4 : "c"        ,  5 : "b"             , 11 : "e", 
         12 : "$\\nu_e$" , 13 : "$\mu$"         , 14 : "$\\nu_{\mu}$", 
         15 : "$\\tau$"  , 16 : "$\\nu_{\\tau}$", 21 : "g", 
         22 : "$\\gamma$"}
 
_leptons = [11, 12, 13, 14, 15, 16]
_charged_leptons = [11, 13, 15]
topMass = 172.5

def PlotTemplate(nevents, lumi):
    Plots = {
                "yTitle" : "Entries (a.u.)",
                "xMin" : 0, 
                "yMin" : 0, 
                "xMax" : None,
                "xBins" : None,
                "OutputDirectory" : "./Figures/", 
                "Style" : "ATLAS",
                "ATLASLumi" : lumi,
                "NEvents" : nevents
            }
    return Plots


def DileptonAnalysis(Ana):

    ReconstructedHadTopMass = {"Res": [], "Spec": []}
    ReconstructedLepTopMass = {"Res": [], "Spec": []}
    ReconstructedResonanceMass = []

    nevents = 0
    neventsNotPassed = 0
    lumi = 0
    for ev in Ana:

        print("---New event---")
        
        event = ev.Trees["nominal"]
        nevents += 1
        lumi += event.Lumi
    
        lquarks = []
        bquarks = []
        leptons = []
        for p in event.TopChildren:
            if abs(p.pdgid) < 5:
                lquarks.append(p)
            elif abs(p.pdgid) == 5:
                bquarks.append(p)
            elif abs(p.pdgid) in _charged_leptons:
                leptons.append(p)
        print(f"Number of light quarks: {len(lquarks)}")
        print(f"Number of b quarks: {len(bquarks)}")
        print(f"Number of leptons: {len(leptons)}")
        print(f"Charge of leptons: {[l.charge for l in leptons]}")

        if len(leptons) != 2 or leptons[0].charge != leptons[1].charge or len(bquarks) != 4 or len(lquarks) != 4:
            neventsNotPassed += 1
            continue
        
        # Match leptons with their closest b quarks
        closestPairs = []
        while leptons != []:
            lowestDR = 100
            for il, l in enumerate(leptons):
                print(f"lepton {il}")
                for ib, b in enumerate(bquarks):
                    print(f"b-quark {ib}")
                    print(f"dR = {l.DeltaR(b)}")
                    if l.DeltaR(b) < lowestDR: # Possibility of adding requirement: if np.sign(b.charge) != np.sign(l.charge)
                        print("Closest so far")
                        lowestDR = l.DeltaR(b)
                        closestB = b
                        closestL = l
            print("Removing last closest lepton and b from lists and adding them to closestPairs")
            leptons.remove(closestL)
            bquarks.remove(closestB)
            closestPairs.append([closestB, closestL])
        closestLeptonicGroup = sum(closestPairs[0])
        remainingLeptonicGroup = sum(closestPairs[1])
        if closestLeptonicGroup.pt > remainingLeptonicGroup.pt:
            print("Pair with smallest dR has largest pT")
            LeptonicResTop = closestLeptonicGroup
            LeptonicSpecTop = remainingLeptonicGroup
        else:
            print("Remaining pair has largest pT")
            LeptonicResTop = remainingLeptonicGroup
            LeptonicSpecTop = closestLeptonicGroup

        # Find the group of one b quark and two jets for which the invariant mass is closest to that of a top quark
        closestGroups = []
        while bquarks != []:
            lowestError = 1e100
            for ib, b in enumerate(bquarks):
                print(f"b-quark {ib}")
                for ipair, pair in enumerate(combinations(lquarks, 2)):
                    print(f"light quark pair {ipair}")
                    IM = sum([b, pair[0], pair[1]]).CalculateMass()
                    print(f"IM = {IM}")
                    if abs(topMass - IM) < lowestError:
                        bestB = b
                        bestQuarkPair = pair
                        lowestError = abs(topMass - IM)
                        print("Lowest error so far")
            print("Removing last closest group from lists and adding them to closestGroups")
            bquarks.remove(bestB)
            lquarks.remove(bestQuarkPair[0])
            lquarks.remove(bestQuarkPair[1])
            closestGroups.append([bestB, bestQuarkPair[0], bestQuarkPair[1]])
      
        bestHadronicGroup = sum(closestGroups[0])
        remainingHadronicGroup = sum(closestGroups[1])
        if bestHadronicGroup.pt > remainingHadronicGroup.pt:
            print("Group with closest mass to top has largest pT")
            HadronicResTop = bestHadronicGroup
            HadronicSpecTop = remainingHadronicGroup
        else:
            print("Remaining group has largest pT")
            HadronicResTop = remainingHadronicGroup
            HadronicSpecTop = bestHadronicGroup

        print(f"Hadronic top mass: res = {HadronicResTop.CalculateMass()}, spec = {HadronicSpecTop.CalculateMass()}")
        print(f"Leptonic top mass: res = {LeptonicResTop.CalculateMass()}, spec = {LeptonicSpecTop.CalculateMass()}")
        print(f"Resonance mass: {sum([HadronicResTop, LeptonicResTop]).CalculateMass()}")
        ReconstructedHadTopMass["Res"].append(HadronicResTop.CalculateMass())
        ReconstructedHadTopMass["Spec"].append(HadronicSpecTop.CalculateMass())
        ReconstructedLepTopMass["Res"].append(LeptonicResTop.CalculateMass())
        ReconstructedLepTopMass["Spec"].append(LeptonicSpecTop.CalculateMass())
        ReconstructedResonanceMass.append(sum([HadronicResTop, LeptonicResTop]).CalculateMass())

    print(f"Number of events not passed: {neventsNotPassed} / {nevents}")

    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Reconstructed Hadronic Top Mass"
    Plots["xTitle"] = "Mass (GeV)"
    Plots["xBins"] = 200
    Plots["xMin"] = 0
    Plots["xMax"] = 200
    Plots["Filename"] = "RecoHadTopMass"
    Plots["Histograms"] = []

    for i in ReconstructedHadTopMass:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = ReconstructedHadTopMass[i]
        Plots["Histograms"].append(TH1F(**_Plots))
    
    X = CombineTH1F(**Plots)
    X.SaveFigure()

    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Reconstructed Leptonic Top Mass"
    Plots["xTitle"] = "Mass (GeV)"
    Plots["xBins"] = 200
    Plots["xMin"] = 0
    Plots["xMax"] = 200
    Plots["Filename"] = "RecoLepTopMass"
    Plots["Histograms"] = []

    for i in ReconstructedLepTopMass:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = ReconstructedLepTopMass[i]
        Plots["Histograms"].append(TH1F(**_Plots))
    
    X = CombineTH1F(**Plots)
    X.SaveFigure()

    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Reconstructed Resonance Mass"
    Plots["xTitle"] = "Mass (GeV)"
    Plots["xBins"] = 150
    Plots["xMin"] = 0
    Plots["xMax"] = 1500
    Plots["Filename"] = "RecoResMass"
    Plots["Histograms"] = []
    _Plots = {}
    _Plots["xData"] = ReconstructedResonanceMass
    Plots["Histograms"].append(TH1F(**_Plots))
    
    X = CombineTH1F(**Plots)
    X.SaveFigure()



    



 

