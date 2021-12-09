from torch_scatter import scatter
import torch

src = torch.randn(2, 6, 4)
index = torch.tensor([0, 1, 0, 1, 2, 1])

tmp = torch.zeros(max(index)+1, 4)
print(tmp)

print(src[0], index)

for i in range(len(src[0])):
    tmp[index[i]] += src[0][i]

#tmp[index] += src[0]
#tmp[index] += src[1][index]

print(tmp)
print("")



print(src)
out = scatter(src, index, dim = 1, reduce = "sum")
print("")
print(out)
print(out.size())


from skhep.math.vectors import LorentzVector

# e, pt, eta, phi
p1 = [ 1.5530e+05,  1.5401e+05, -1.2957e-01, -2.3062e+00]

P_v1 = LorentzVector()
P_v1.setptetaphie(p1[1], p1[2], p1[3], p1[0])

P_v2 = LorentzVector()
P_v2.setptetaphie(p1[1], p1[2], p1[3], p1[0])
tot = P_v2 + P_v1

p2 = [2*i for i in p1[:3]]
p2.append(p1[3])
print(p2)
P_p2 = LorentzVector()
P_p2.setptetaphie(p2[1], p2[2], p2[3], p2[0])



print(P_p2.mass, tot.mass)










