from AnalysisTopGNN import Analysis 
from AnalysisTopGNN.Events import Event
from AnalysisTopGNN.IO import PickleObject, UnpickleObject

#direc = "/home/tnom6927/Downloads/samples/tttt/QU_0.root"
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

ev = UnpickleObject("TMP")
singlelepton = [i for i in ev.TopChildren if i.Parent[0].DecayLeptonically()]
singlelepton = {abs(i.pdgid) : i for i in singlelepton}
b = singlelepton[5]
muon = singlelepton[13]
nu = singlelepton[14]

mW = 80.385*1000 # MeV : W Boson Mass
mT = 172.5*1000  # MeV : t Quark Mass
mN = 0           # GeV : Neutrino Mass

from BaseFunctionsTests import *

print("---- Comparing the Four Vectors ----")
print("-> b-quark")
TestFourVector(b)
print("-> muon")
TestFourVector(muon)
print("-> neutrino")
TestFourVector(nu)

print("---- Comparing CosTheta and SinTheta ----")
TestCosTheta(b, muon)

print("---- Comparing x0 ----")
print("b + W:")
Testx0(mT, mW, b)

print("nu + mu:")
Testx0(mW, mN, muon)

print("----- Comparing Beta -----")
print("b")
TestBeta(b)

print("muon")
TestBeta(muon)







