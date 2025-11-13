from atomics import *
from particle import *

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
