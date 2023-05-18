from AnalysisG import Analysis 
from AnalysisG.Events import Event 
from AnalysisG.IO import UnpickleObject, PickleObject

from Studies.Resonance.ZPrimePtMass import ZPrimeMatrix
import PlottingCode.Resonance_ZPrimePtMass as ZPrime

import Studies.Resonance.ResonanceTruthTops as RTT_Sel
import PlottingCode.ResonanceTruthTops as RTT_Plot

import Studies.Resonance.ResonanceTruthChildren as RTC_Sel
import PlottingCode.ResonanceTruthChildren as RTC_Plot

import Studies.Resonance.ResonanceFromTruthJets as RTJ_Sel
import PlottingCode.ResonanceFromTruthJets as RTJ_Plot

import Studies.Resonance.ResonanceFromJets as RJJ_Sel
import PlottingCode.ResonanceFromJets as RJJ_Plot

import Studies.TruthTops.TopDecay as TTT_Sel
import PlottingCode.TopDecay as TTT_Plot

import Studies.TruthTops.TopsFromTruthJets as TTJ_Sel
import PlottingCode.TopsFromTruthJets as TTJ_Plot

import Studies.TruthTops.TopsFromJets as TJ_Sel
import PlottingCode.TopsFromJets as TJ_Plot

import Studies.TruthChildren.TruthChildrenKinematics as TCK_Sel
import PlottingCode.TruthChildrenKinematics as TCK_Plot

import Studies.Event.TruthEvent as ETE_Sel
import PlottingCode.TruthEvent as ETE_Plot

import Studies.Event.EventNeutrino as EN_Sel
import PlottingCode.EventNeutrino as EN_Plot
import os 
import shutil


toRun = [
        #"ZPrimeMatrix", 
        #"ResonanceDecayModes", 
        #"ResonanceMassFromTops", 
        #"ResonanceDeltaRTops", 
        #"ResonanceTopKinematics", 
        #"EventNTruthJetAndJets", 
        #"EventMETImbalance",
        #"TopDecayModes", 
        #"ResonanceMassFromChildren", 
        #"DeltaRChildren", 
        #"Kinematics", 
        #"EventNuNuSolutions", 
        #"ResonanceMassTruthJets", 
        #"ResonanceMassTruthJetsNoSelection", 
        #"TopMassTruthJets", 
        #"TopTruthJetsKinematics", 
        #"ResonanceMassJets", 
        "TopMassJets", 
]

studies = {
            "ZPrimeMatrix" : ZPrimeMatrix,
            "ResonanceDecayModes" : RTT_Sel.ResonanceDecayModes, 
            "ResonanceMassFromTops" : RTT_Sel.ResonanceMassFromTops, 
            "ResonanceDeltaRTops" : RTT_Sel.ResonanceDeltaRTops, 
            "ResonanceTopKinematics" : RTT_Sel.ResonanceTopKinematics,
            "EventNTruthJetAndJets" : ETE_Sel.EventNTruthJetAndJets, 
            "EventMETImbalance" : ETE_Sel.EventMETImbalance,
            "TopDecayModes" : TTT_Sel.TopDecayModes, 
            "ResonanceMassFromChildren" : RTC_Sel.ResonanceMassFromChildren,
            "DeltaRChildren" : TCK_Sel.DeltaRChildren,
            "Kinematics" : TCK_Sel.Kinematics, 
            "EventNuNuSolutions" : EN_Sel.EventNuNuSolutions, 
            "ResonanceMassTruthJets" : RTJ_Sel.ResonanceMassTruthJets, 
            "ResonanceMassTruthJetsNoSelection" : RTJ_Sel.ResonanceMassTruthJetsNoSelection, 
            "TopMassTruthJets" : TTJ_Sel.TopMassTruthJets, 
            "TopTruthJetsKinematics" : TTJ_Sel.TopTruthJetsKinematics, 
            "ResonanceMassJets" : RJJ_Sel.ResonanceMassJets, 
            "TopMassJets" : TJ_Sel.TopMassJets, 
}

studiesPlots = {
            "ZPrimeMatrix" : ZPrime.Plotting,
            "ResonanceDecayModes" : RTT_Plot.ResonanceDecayModes, 
            "ResonanceMassFromTops" : RTT_Plot.ResonanceMassFromTops, 
            "ResonanceDeltaRTops" : RTT_Plot.ResonanceDeltaRTops, 
            "ResonanceTopKinematics" : RTT_Plot.ResonanceTopKinematics,
            "EventNTruthJetAndJets" : ETE_Plot.EventNTruthJetAndJets,
            "EventMETImbalance" : ETE_Plot.EventMETImbalance,
            "TopDecayModes" : TTT_Plot.TopDecayModes,
            "ResonanceMassFromChildren" : RTC_Plot.ResonanceMassFromChildren, 
            "DeltaRChildren" : TCK_Plot.DeltaRChildren,
            "Kinematics" : TCK_Plot.Kinematics, 
            "EventNuNuSolutions" : EN_Plot.EventNuNuSolutions, 
            "ResonanceMassTruthJets" : RTJ_Plot.ResonanceMassTruthJets,
            "ResonanceMassTruthJetsNoSelection" : RTJ_Plot.ResonanceMassTruthJetsNoSelection, 
            "TopMassTruthJets" : TTJ_Plot.TopMassTruthJets, 
            "TopTruthJetsKinematics" : TTJ_Plot.TopTruthJetsKinematics, 
            "ResonanceMassJets" : RJJ_Plot.ResonanceMassJets, 
            "TopMassJets" : TJ_Plot.TopMassJets, 
}

Masses = ["1000", "900", "800", "700", "600", "500", "400"]
Topo = ["Dilepton", "SingleLepton"]
smpl = os.environ["Samples"]
os.makedirs("./FigureCollection", exist_ok = True)

SampleNames = {}
for topo in Topo:
    for m in Masses:
        smpls = "" #"/" + topo + "Collection/MadGraphPythia8EvtGen_noallhad_"
        SampleNames |= {"BSM-4t-" + ("SL" if topo == "SingleLepton" else "DL") + "-" + m : smpl + "ttH_tttt_m" + m}

it = 1
for smpl in SampleNames:
    Ana = Analysis()
    for i in toRun: Ana.AddSelection(i, studies[i])
    for i in toRun: Ana.MergeSelection(i)
    
    Ana.ProjectName = "_ProjectL"
    Ana.InputSample(smpl, SampleNames[smpl])
    Ana.Event = Event 
    Ana.EventStop = 1000
    Ana.Threads = 12
    Ana.chnk = 1000
    Ana.EventCache = True
    Ana.PurgeCache = False
    Ana.Launch
    
    # Runs the plotting code
    for i in toRun:
        x = UnpickleObject(Ana.ProjectName + "/Selections/Merged/" + i + ".pkl")
        print("Making Plots for: " + i)
        studiesPlots[i](x)
    
    print("Finished: " + smpl + ": " + str(it) + "/" + str(len(SampleNames)))
    shutil.move("./Figures", "./FigureCollection/" + smpl)
    it += 1
    

