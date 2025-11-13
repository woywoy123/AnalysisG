from atomics import *
from particle import *
from conuic import *

class Conuic:

    def __init__(self, met, phi, detector, event = None):

        self.fig = None #figure()
        #self.fig.auto_lims = True

        #self.fig.plot_title(f'Event Ellipses {event.idx}', 12)
        #self.fig.axis_label("x", "Sx")
        #self.fig.axis_label("y", "Sy")
        #self.fig.axis_label("z", "Sz")

        self.px = math.cos(phi)*met
        self.py = math.sin(phi)*met
        self.pz = 0
        self.loss = 0

        self.lep, self.jet = [], []
        for i in detector:
            l = self.lep if i.mass < 200 else self.jet
            l.append(i)

        self.engine = [conuic(i, j, event, self.fig) for i in self.lep for j in self.jet]

        _tmp_lep = {}
        for i in self.engine:
            if i.error is None: self.loss += i.is_truth; continue
            try: _tmp_lep[i.lep.hash][i.jet.hash] = i.jet
            except KeyError: _tmp_lep[i.lep.hash] = {i.lep.hash : i.lep}

        print("--------- second ------------")
        self.engine = []
        for i in _tmp_lep:
            jets = [_tmp_lep[i][k] for k in _tmp_lep[i] if k != i]
            if not len(jets): continue
            jets = sum(jets) if len(jets) > 1 else jets[0]
            c = conuic(_tmp_lep[i][i], jets, event, self.fig)
            print(c.error, c.is_truth)
            if c.error is not None: self.engine.append(c)
            elif c.is_truth: print("---- rejected truth !!!!!!!!!!!!!!!---- ") 
       
        print(self.loss)
        #self.fig.show()



class event:

    def __init__(self, idx):
        self.cache = []
        self.met = 0
        self.phi = 0
        self.idx = idx
        self.DetectorObjects = {}
        self.truth_pairs = {}
        self.truth = []

    def build(self):
        self.met, self.phi = get_numbers(self.cache[0].split(" "))
        blocks = {}
        title = None
        for i in self.cache:
            if "Truth"    in i: title = "neutrino"
            if "leptons"  in i: title = "leptons"  
            if "jets"     in i: title = "jets"
            if "Detector" in i: title = "detector"
            if title is None: continue
            if title not in blocks: blocks[title] = []; continue;
            lx = []
            l = i.split(" ")
            for k in l: lx += [k if ":" not in k else k.split(":")[-1]]
            c = get_numbers(lx) 
            if len(c) < 1: continue
            blocks[title] += [c]

        self._detector(blocks["detector"])
        self._assign(blocks["jets"])
        self._assign(blocks["leptons"])
        self._neutrino(blocks["neutrino"])
        self._truthpair()
        self.cache = []

    def _detector(self, blk):
        for i in range(len(blk)):
            p = Particle(*blk[i])
            self.DetectorObjects[p.hash] = p

    def _assign(self, blk):
        for i in range(len(blk)):
            hash, ti = blk[i]
            if hash not in self.DetectorObjects: continue
            self.DetectorObjects[hash].top_index = ti
    
    def _neutrino(self, blk):
        for i in range(len(blk)):
            p = Particle(*blk[i][:-1])
            p.top_index = blk[i][-1]
            self.truth.append(p)
            self.truth_pairs[p.top_index] = [p]

    def _truthpair(self):
        for i in self.DetectorObjects.values():
            if i.top_index == -1: continue
            if i.top_index not in self.truth_pairs: continue
            self.truth_pairs[i.top_index] += [i]

    def __str__(self):
        o = "_____________ Detector Objects __________ \n"
        for i in self.DetectorObjects: o += self.DetectorObjects[i].__str__() + "\n"
        o += "_____________ MET --------------- \n"
        o += string(self, "met") + " " + string(self, "phi") + "\n"
        o += "*********** TRUTH NEUTRINO ************* \n"
        for i in self.truth: o += i.__str__() + "\n"
        return o

class DataLoader:

    def __init__(self, pth = "./data.txt"):
        self.events = {}
        f = open(pth).readlines()

        idx = -1
        for i in f:
            data = i.split("\n")[0]
            if "new event" in data: idx += 1; continue
            if idx not in self.events: self.events[idx] = event(idx)
            self.events[idx].cache += [data]
        for i in self.events: self.events[i].build()
    
    def __iter__(self): 
        self.itr = iter(self.events.values())
        return self

    def __next__(self):
        return next(self.itr)
