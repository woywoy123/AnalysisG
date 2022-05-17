import torch
import LorentzVector as LV
import math

from skhep.math.vectors import LorentzVector 

# Test Four vector
# pt, eta, phi, energy
v = [139742.28, -1.4921323, -2.0837228, 326400.22]

# ===== Test if device works ===== #
x_cuda = LV.ToPxPyPzE(v[0], v[1], v[2], v[3], "cuda")
x_cpu  = LV.ToPxPyPzE(v[0], v[1], v[2], v[3], "cpu")

assert str(x_cuda.device) == "cuda:0"
assert str(x_cpu.device) == "cpu"

# ===== Test if the implementation reproduces the px, py, pz, energy vector 
p = LorentzVector()
p.setptetaphie(v[0], v[1], v[2], v[3])

p_cpu = torch.round(x_cpu, decimals = 0).tolist()
try:
    assert p_cpu[0] == round(p.px)
    assert p_cpu[1] == round(p.py)
    assert p_cpu[2] == round(p.pz)
    assert p_cpu[3] == round(p.e)

except:
    print("----> Failed!")
    print(p_cpu)
    print([p.px, p.py, p.pz, p.e])

v = [[139742.28, -1.4921323, -2.0837228, 326400.22], 
     [139742.28, -1.4921323, -2.0837228, 326400.22],
     [139742.28, -1.4921323, -2.0837228, 326400.22]]

x_list = LV.ListToPxPyPzE(v, "cuda")
print(p.mass, math.sqrt(sum([-p.px**2, -p.py**2, -p.pz**2, p.e**2])))

v1 = LV.MassFromPxPyPzE(x_list)
print(v1)

v2 = LV.MassFromPxPyPzE(x_cpu)
print(v2)

v3 = LV.MassFromPxPyPzE(torch.tensor([p.px, p.py, p.pz, p.e]))
print(v3)

c = torch.tensor(v)
print(LV.MassFromPtEtaPhiE(torch.tensor(v, device = "cuda")))
