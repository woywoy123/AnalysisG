from AnalysisG.core import IO
import pathlib
import pickle

class SampleContent:

    def __init__(self):
        self._isdata = False
        self._kfold = None
        self._epoch = None
        self._mrk = None
        self._trig = False
        self._samples = {}

    def addSample(self, inpt):
        lst = inpt.split("/")
        fname = lst[-1]
        smpln = lst[-2]

        io = IO(inpt)
        io.Trees = ["nominal"]
        io.Leaves = ["event_weight"] if not self._isdata else ["weight_mc"]
        lx = len(io)

        try: self._samples[smpln][fname] = lx
        except KeyError:
            self._samples[smpln] = {}
            self._samples[smpln][fname] = lx

        if self._trig or self._isdata: return
        if self._kfold is None: self._kfold = lst[-3]
        if self._epoch is None: self._epoch = lst[-4]
        if self._mrk is None: self._mrk = lst[-5]
        self._trig = True

def build_samples(pth):
    return [str(i) for i in pathlib.Path(pth).glob("**/*.root") if str(i).endswith(".root")]

try: data = pickle.load(open("./cache/data.pkl", "rb"))
except: data = {"null" : {}}


#for i in data:
    #    if i == "mc16": continue
#    if i == "null": continue
#    print(data[i]._samples)
#exit()


root = "/CERN/Samples/mc16-full/"
mc16_samples = build_samples(root)
data["mc16"] = {}
for i in mc16_samples:
    smpls = i.split("/")[-2] + "/" + i.split("/")[-1]
    try: dt = data["mc16"][smpls]; continue
    except KeyError:
        dt = SampleContent()
        dt._isdata = True
        data["mc16"][smpls] = dt
    dt.addSample(i)
    update = True
    print(i)

#gnn = "/CERN/trainings/mc16-full-inference/ROOT/GraphJets_bn_1_Grift/"
#gnn = "/CERN/trainings/mc16-full-inference/ROOT/GraphJets_bn_1_Grift/"
#gnn_samples = build_samples(gnn)
#lxn = len(gnn_samples)
#
#update = False
#for i in range(lxn):
#    name = gnn_samples[i]
#    try: data["null"][name]; continue
#    except KeyError: pass
#
#    print(name, float(i / lxn))
#    smpln = "/".join(name.split("/")[:-3])
#    try: dt = data[smpln]
#    except KeyError:
#        dt = SampleContent()
#        data[smpln] = dt
#    dt.addSample(name)
#    data["null"][name] = True
#    update = True
#    if i % 1000 != 1000-1: continue
#    pathlib.Path("./cache").mkdir(parents = True, exist_ok = True)
#    f = open("./cache/data.pkl", "wb")
#    pickle.dump(data, f)
#    f.close()
#    update = False
#
if update:
    pathlib.Path("./cache").mkdir(parents = True, exist_ok = True)
    f = open("./cache/data.pkl", "wb")
    pickle.dump(data, f)
    f.close()


