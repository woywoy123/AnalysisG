from classes import * 
from conuic import Conuic
 
for i in DataLoader():
    if i.idx != 22: continue
    print("EVENT: ", i.idx)
    nu = Conuic(i.met, i.phi, list(i.DetectorObjects.values()), i)
    print("\n\n")
    exit()
