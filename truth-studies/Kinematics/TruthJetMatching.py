from AnalysisTopGNN.Plotting import TH1F, CombineTH1F, TH2F
from copy import copy

PDGID = { 1 : "d"        ,  2 : "u"             ,  3 : "s", 
          4 : "c"        ,  5 : "b"             , 11 : "e", 
         12 : "$\\nu_e$" , 13 : "$\mu$"         , 14 : "$\\nu_{\mu}$", 
         15 : "$\\tau$"  , 16 : "$\\nu_{\\tau}$", 22 : "g"}

CounterPDGID = {"d"            : 0, "u"       : 0, "s"              : 0, "c"    : 0, 
                "b"            : 0, "e"       : 0, "$\\nu_e$"       : 0, "$\mu$": 0, 
                "$\\nu_{\mu}$" : 0, "$\\tau$" : 0, "$\\nu_{\\tau}$" : 0, "g"    : 0}
 
_leptons = [11, 12, 13, 14, 15, 16]

def PlotTemplate(nevents, lumi):
    Plots = {
                "yTitle" : "Entries (a.u.)",
                "xMin" : 0, 
                "yMin" : 0, 
                "xMax" : None,
                "xBins" : None,
                "OutputDirectory" : "./Figures/TruthJet", 
                "Style" : "ATLAS",
                "ATLASLumi" : lumi,
                "NEvents" : nevents
            }
    return Plots

def TruthJetPartons(Ana):
    TopChildrenPDGID = copy(CounterPDGID)
    
    nevents = 0
    lumi = 0
    tops = 0
    for ev in Ana:
        event = ev.Trees["nominal"]
        nevents += 1
        lumi += event.Lumi
 

