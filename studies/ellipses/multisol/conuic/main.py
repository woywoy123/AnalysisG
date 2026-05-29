from conuix.base.dataloader import * 
from conuix.conuic import * 
from original import *

def run(one = True):
    for i in DataLoader():
#        if i.idx != 40: continue
        print("_________________" + str(i.idx) + "_____________")
        for j in i.top:
            if i.lepton[j].skp or i.neutrino[j].skp: continue
            p = NuSol(i.bquark[j], i.lepton[j], i.neutrino[j], False)
            n = NuSol(i.bquark[j], i.lepton[j], i.neutrino[j], True)
            conuic(i.bquark[j], i.lepton[j], i.neutrino[j], p, n)
            if one: return 

if __name__ == "__main__":
    run(False)
