from AnalysisTopGNN import Analysis 
from AnalysisTopGNN.Events import Event
from AnalysisTopGNN.IO import PickleObject, UnpickleObject

#direc = "/home/tnom6927/Downloads/ttH_tttt_m1000/output.root"
#Ana = Analysis()
#Ana.InputSample("bsm1000", direc)
#Ana.Event = Event
#Ana.EventCache = True
#Ana.DumpPickle = True 
#Ana.Launch()
#
#
#for i in Ana:
#    ev = i.Trees["nominal"]
#    
#    it = 0
#    for t in ev.Tops:
#        it += 1 if t.DecayLeptonically() else 0
#    if it == 1:
#        PickleObject(ev, "TMP")
#        break
#
ev = UnpickleObject("TMP")
singlelepton = [i for i in ev.TopChildren if i.Parent[0].DecayLeptonically()]

from neutrino_momentum_reconstruction_python3 import singleNeutrinoSolution as sNS
import ROOT as r
from PhysicsCPU import ToPx, ToPy, ToPz
import numpy as np
from Reimplementation import *





met = ev.met 
phi = ev.met_phi

#met_x = ToPx(met, phi)
#met_y = ToPy(met, phi)

b = singlelepton[0]
nu = singlelepton[1]
muon = singlelepton[2]

#TestNuSolutionSteps(b, muon)
TestSingleNeutrinoSolutionSegment(b, muon, ToPx(met, phi), ToPy(met, phi), np.array([[2, 1], [1, 2]]))


#sigma = np.array([[1000, 200], [200, 1000]])
#sn = sNS(b_pmc, muon_pmc, met_x, met_y, sigma)
#print("-> Prediction: ", sn.solutions)
#print("-> Cartesian Truth of Neutrinos: ", ToPx(nu.pt, nu.phi), ToPy(nu.pt, nu.phi), ToPx(nu.pt, nu.eta))
#print("-> Pseudo-Rapidity: ", nu.pt, nu.phi, nu.eta)
#print("-> Mass of Top Quark: ", sum(singlelepton).CalculateMass())
