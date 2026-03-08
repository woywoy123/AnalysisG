from particle import *
from variables import *
from atomics import *

class Conuix:
    def __init__(self, lep, jet, event):
        self.lep, self.jet = lep, jet
        self.truth_pair = []
        self.is_truth  = True
        self.is_truth *= (jet.top_index == lep.top_index)
        self.is_truth *= (lep.top_index in event.truth_pairs)
        self.truth_pair = event.truth_pairs[jet.top_index] if self.is_truth else []
        if not self.is_truth: return 
        i = Particle(0, 0, 0, 0)
        for j in self.truth_pair:
            if self.lep.hash == j.hash: continue
            if self.jet.hash == j.hash: continue
            i = j
            break
        algorithm(jet, lep, i)


class Conuic:
    def __init__(self, particles, event):
        self.lep, self.jet = [], []
        for i in particles:
            l = self.lep if i.mass < 200 else self.jet
            l.append(i)
        self.engine = [Conuix(i, j, event) for i in self.lep for j in self.jet]
 
