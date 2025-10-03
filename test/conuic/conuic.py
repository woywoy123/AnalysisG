from visualize import *
from atomics import *
from cache import *
from eigen import *
from debug import *
from poly import *

import numpy as np
import math

class conuic(matrix, traject, eigen):

    def __init__(self, lep, bqrk, event_t = None, runtime = None):
        traject.__init__(self, runtime)
        matrix.__init__(self)
        eigen.__init__(self)

        self.z    = 1
        self.tau  = 0
        self.m_nu = 0 #1e-12

        self.lep = lep
        self.jet = bqrk
        self.cache()
        self.truth_pair = []
   
        self.is_truth = False
        if event_t is None: return 
        if self.jet.top_index != self.lep.top_index: return 
        if self.lep.top_index not in event_t.truth_pairs: return 
        self.truth_pair = event_t.truth_pairs[self.jet.top_index]
        self.is_truth = True

    def Sx(self, z, t): return p_Sx1(self, z, t)
    def Sy(self, z, t): return p_Sy1(self, z, t)
    
    def H_tilde(self, z, t):  return p_h_tilde(self, z, t)
    def H_matrix(self, z, t): return p_hmatrix(self, z, t)
    def Z2(self, Sx, Sy):     return p_Z2(self, Sx, Sy)











class Conuic(debug):

    def __init__(self, met, phi, detector, event = None):
        debug.__init__(self)
        self.debug_mode = True
        self.fig = figure()
#        self.fig.auto_lims = False
#        self.fig.max_x = 1
#        self.fig.max_y = 1
#        self.fig.max_z = 1
#
#        self.fig.min_x = -1
#        self.fig.min_y = -1
#        self.fig.min_z = -1
#

        self.fig.plot_title(f'Event Ellipses {event.idx}', 12)
        self.fig.axis_label("x", "Sx")
        self.fig.axis_label("y", "Sy")
        self.fig.axis_label("z", "Sz")

        self.px = math.cos(phi)*met
        self.py = math.sin(phi)*met
        self.pz = 0
        self.lep, self.jet = [], []
        for i in detector:
            l = self.lep if i.mass < 200 else self.jet
            l.append(i)
            i.gev

        self.engine = [conuic(i, j, event, self.fig) for i in self.lep for j in self.jet]
        for i in range(len(self.engine)*self.debug_mode): self.debug(i); exit()
        
        self.fig.add_object("ellipse-1", self.engine[0])
        #self.fig.add_object("ellipse-2", self.engine[1])
        self.fig.show()
