import torch
import LorentzVector as LV
import math

from skhep.math.vectors import LorentzVector 

x = LV.ToPxPyPzE(0.1, 0.1, 0.1, 1, "cuda")
print(x.device)
print(x)

print(LV.GetMass(x))

v = [139742.28, -1.4921323, -2.0837228, 326400.22]
x = LV.ToPxPyPzE(v[0], v[1], v[2], v[3], "cpu")
print(x)
print(LV.GetMass(x))

print(x.device)
print(x)




p = LorentzVector()
p.setptetaphie(0.1, 0.1, 0.1, 1)
print(p.px, p.py, p.pz, p.e)
print(p.mass)

