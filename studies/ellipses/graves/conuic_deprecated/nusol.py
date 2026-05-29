from classes import * 
from conuic import Conuic

ix = 0
idx = 1

loss  = {}
truth = {}
junk  = {}
njunk = {}
for i in DataLoader():
#    if i.idx != idx: continue
    print("EVENT: ", i.idx)
    nu = Conuic(i.met, i.phi, list(i.DetectorObjects.values()), i)
    print("\n\n")
    print(ix)

    try: truth[nu.ntruth] += [nu.ntruth - nu.loss]
    except KeyError: truth[nu.ntruth] = [nu.ntruth - nu.loss]

    try: junk[nu.ntruth] += [nu.fake]
    except KeyError: junk[nu.ntruth] = [nu.fake]

    try: njunk[nu.ntruth] += [nu.nfake]
    except KeyError: njunk[nu.ntruth] = [nu.nfake]

    if ix > 10: break
    ix += 1
    #idx += 1
    break

#for i in sorted(truth):
#    print(
#            "multiplicity: ", i, 
#            "expected: "    , i * len(truth[i]), 
#            "recovered: "   , sum(truth[i]), 
#            "recovery (%): ", 100 * float(sum(truth[i])) / float(i * len(truth[i]) if len(truth[i]) and i > 0 else 1),
#            "fakes (%): "   , 100 * float(sum(junk[i])) / float(sum(njunk[i]))
#    )
