from matrix import *
from atomics import *
from debug import *
import numpy as np

class conuic(matrix, debug):

    def __init__(self, lep, bqrk, event_t = None, runtime = None):
        self.m_nu = 0 
        self.lep = lep
        self.jet = bqrk

        matrix.__init__(self)
        debug.__init__(self)
        self.truth_pair = []
        if event_t is None: return 
        self.is_truth = False
        if self.jet.top_index != self.lep.top_index: return self.debug()
        if self.lep.top_index not in event_t.truth_pairs: return self.debug()
        self.truth_pair = event_t.truth_pairs[self.jet.top_index]
        self.is_truth = True
        self.debug()
    

