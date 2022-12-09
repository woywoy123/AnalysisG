from AnalysisTopGNN.IO import UnpickleObject
from PhysicsCPU import *

ev = UnpickleObject("TMP")
singlelepton = [i for i in ev.TopChildren if i.Parent[0].DecayLeptonically()]
b = singlelepton[0]
nu = singlelepton[1]
muon = singlelepton[2]
 
print(ToPx(b.pt, b.phi, "cuda"))

