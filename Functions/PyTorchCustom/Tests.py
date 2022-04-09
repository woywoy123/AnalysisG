import torch
import LorentzVector as LV

from skhep.math.vectors import LorentzVector 

x = LV.ToPxPyPzE(0.1, 0.1, 0.1, 1, "cuda")
print(x.device)
print(x)

print(LV.GetMass(x))



print(x.device)
print(x)




p = LorentzVector()
p.setptetaphie(0.1, 0.1, 0.1, 1)
print(p.px, p.py, p.pz, p.e)
print(p.mass)

