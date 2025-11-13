from classes import * 

for i in DataLoader():
    print("EVENT: ", i.idx)
    nu = Conuic(i.met, i.phi, list(i.DetectorObjects.values()), i)
    exit()
