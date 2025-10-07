from classes import * 
from conuic import Conuic

ix = 0
idx = 22
for i in DataLoader():
    if i.idx != idx: continue
    print("EVENT: ", i.idx)
    nu = Conuic(i.met, i.phi, list(i.DetectorObjects.values()), i)
    print("\n\n")
    print(ix)
    if ix > 10: break
    ix += 1
    #idx += 1
