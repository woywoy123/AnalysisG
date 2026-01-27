from classes import * 
from time import sleep 


for i in DataLoader():
#    if i.idx != 1: continue
    print("====================== EVENT: ", i.idx, " ==========================")
    nu = Conuic(i.met, i.phi, list(i.DetectorObjects.values()), i)
#    exit()
    sleep(0.5)
