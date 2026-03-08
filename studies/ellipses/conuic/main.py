from dataloader import * 
from conuic import *

for i in DataLoader():
    #    if i.idx != 41: continue
    print("====================== EVENT: ", i.idx, " ==========================")
    nu = Conuic(list(i.DetectorObjects.values()), i)
