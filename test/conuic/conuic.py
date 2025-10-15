from particle import *
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

        self.l    = 1 # lambda 
        self.z    = 1 # scaling factor Z
        self.tau  = 1 # hyperbolic variable
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

    # NOTE: all of the below are defined in eigen
    # This space is not designed for computational performance
    # Here we only care about keeping the code readable.
    def P(self, l, z, tau):     return self._P(     l, z, tau)

    # NOTE: derivatives of characteristic polynomial
    def dPdl(self, l, z, tau): return self._dPdL(  l, z, tau)
    def dPdz(self, l, z, tau): return self._dPdZ(  l, z, tau)
    def dPdt(self, l, z, tau): return self._dPdtau(l, z, tau)

    # NOTE: Cases where the derivatives of P w.r.t the variables are 0
    def L0_dPdZ(self, z, tau):   return self._lambda_dPdZ(z, tau)
    def L0_dPdL(self, z, tau):   return self._lambda_dPdL(z, tau)
    def L0_dPdtau(self, z, tau): return self._lambda_dPdtau(z, tau)

    # NOTE: Degenerate roots
    def degenerate_dPdL(self): return self._lambda_dPdL_degenerate()
    def degenerate_dPdZ(self): return self._lambda_dPdZ_degenerate()

    # NOTE: Extremely interesting case dPdtau = 0 and P = 0.
    def MobiusTransform(self, t): return self._M_transform(t)
    def MobiusInverse(self, M):   return self._M_inverse(M)
    def MobiusCoefficients(self): return self._M_coef()
   
    # NOTE: dPdtau = 0 and P = 0, solves this analytically
    # but there are floating point considerations
    def P0_dPdtau0(self, get_all = False): return self._lambda_roots_dPdtau(1, get_all)

    # NOTE: Roots of the characteristic polynomial 
    # where we solve for Z
    def PolyZ(self, l, tau): return self._Z_roots_P(l, tau)


class Conuic(debug):

    def __init__(self, met, phi, detector, event = None):
        debug.__init__(self)
        self.loss   = 0
        self.ntruth = 0
        self.fake   = 0
        self.nfake  = 0

        self.debug_mode = True
        self.fig = figure()
        self.fig.auto_lims = True

        #self.fig.max_x = 4000000
        #self.fig.max_y = 4000000
        #self.fig.max_z = 4000000

        #self.fig.min_x = -1000000
        #self.fig.min_y = -1000000
        #self.fig.min_z = -1000000

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

        self.engine = [conuic(i, j, event, self.fig) for i in self.lep for j in self.jet]
        for i in range(len(self.engine)*self.debug_mode): 
            self.debug(i)
            self.engine[i].scan_regions(0.1)
            self.engine[i].shapes()
            self.fig.add_object("ellip" + str(i), self.engine[i])

        self.candidates = []
        for i in range(len(self.engine)): 
            eg = self.engine[i]
            eg.scan_regions(0.1)
            self.ntruth +=  eg.is_truth
            self.loss   += (eg.reject and eg.is_truth)
            self.fake   += (not eg.reject and not eg.is_truth)
            self.nfake  += (not eg.is_truth)
            self.candidates += [eg]*(not eg.reject)
        
        #for i in range(len(self.candidates)):
        #    for j in range(len(self.candidates)):
        #        if i >= j: continue
        #        self.intersections(self.candidates[i], self.candidates[j])
        self.fig.show()

    def intersections(self, i, j):
        def line_intersection(trgt, r0, d):
            # Projection Coefficients
            beta  = np.dot(d, trgt.A) / np.linalg.norm(trgt.A)**2
            delta = np.dot(d, trgt.B) / np.linalg.norm(trgt.B)**2 
            alpha = np.dot(r0 - trgt.C, trgt.A) / np.linalg.norm(trgt.A)**2
            gamma = np.dot(r0 - trgt.C, trgt.B) / np.linalg.norm(trgt.B)**2 
       
            if len([i for i in [beta, delta, alpha, gamma] if np.isnan(i)]): return [], [], []

            a = beta ** 2 + delta ** 2
            if round(a, 8) == 0: return [],[],[]

            b = 2 * (alpha * beta + gamma * delta)
            c = alpha ** 2 + gamma ** 2 -1
            disc = b**2 - 4 * a * c
            if disc < 0: return [], [], []
            s1 = (-b + np.sqrt(disc))/ (2 * a)
            s2 = (-b - np.sqrt(disc))/ (2 * a)

            phi_v, pts = [], []
            for s in [s1, s2]:
                x = alpha +  beta * s
                y = gamma + delta * s
                px = np.arctan2(y, x)
                if px > 2 * np.pi: px = np.pi + abs(px - np.pi)
                if px < 0: px = 2 * np.pi + px
                phi_v.append(px) 
                pts.append(r0 + s * d)
            return phi_v, pts, [s1, s2]


        def plane_intersection(el1, el2):
            n1, d1, c1 = el1.get_plane()
            n2, d2, c2 = el2.get_plane()

            d11 = np.dot(n1, n1)
            d22 = np.dot(n2, n2)
            d12 = np.dot(n1, n2)

            if round((d11 * d22 - d12**2), 8) == 0: return 

            r0 = (( d1 * d22 - d2 * d12) * n1 + (d2 * d11 - d1 * d12) * n2)
            r0 /= (d11 * d22 - d12**2)
            d = np.cross(n1, n2)
            d = d / np.linalg.norm(d)

            phi1, pts1, s1 = line_intersection(el1, r0, d)
            phi2, pts2, s2 = line_intersection(el2, r0, d)

            if not len(pts1) or not len(pts2): return 
            nu1_1 = el1.hmatrix.dot([np.cos(phi1[0]), np.sin(phi1[0]), 1])
            nu1_2 = el1.hmatrix.dot([np.cos(phi1[1]), np.sin(phi1[1]), 1])

            nu2_1 = el2.hmatrix.dot([np.cos(phi2[0]), np.sin(phi2[0]), 1])
            nu2_2 = el2.hmatrix.dot([np.cos(phi2[1]), np.sin(phi2[1]), 1])
        
            return {
                    "n1" : {"phi" : phi1, "pts" : pts1, "s" : s1, "r0" : r0, "d" : d, "sols" : [nu1_1, nu1_2]}, 
                    "n2" : {"phi" : phi2, "pts" : pts2, "s" : s2, "r0" : r0, "d" : d, "sols" : [nu2_1, nu2_2]}
            }


        # NOTE: work in progress..... 
        # Idea is to find the plane equation of the ellipse
        # then find the line which intersects a pair of ellipses
        # These lines will then be used to further reduce fake contributions
        # The final step is to numerically/analytically solve 
        # For the MET constraint, e.g. met_x = sum_i .....
        # Since H_tilde can be represented as Morbius transformations, 
        # It becomes a polynomial could be solved analytically. 
        # To be seen.
        v0 = np.array([[0, 0, self.px],[0, 0, self.py], [0, 0, self.pz]]) * 0

        traj = traject(self.fig)
        can_i = {abs(k.mobius) : k for k in i.parameters}
        can_j = {abs(t.mobius) : t for t in j.parameters}

        l = 0
        for k in sorted(can_i):
            ox = can_i[k]
            H  = can_i[k].hmatrix
            try: K_ = np.linalg.inv(H).T.dot(circle()).dot(np.linalg.inv(H))
            except: continue
            cX = (v0 - circle()).T.dot(K_).dot(v0 - circle())
            
            E_ = traj.figures(Ellipse, "r" if ox.tag_truth else "b", 1, True)
            traj.ellipse(K_, None, None, E_)
            E_.data.matrix = H
            E_.alpha = 1.0
            
            ox1 = ox
            ox1.hmatrix = H
            break


        for k in sorted(can_j):
            ox = can_j[k]
            H  = can_j[k].hmatrix
            try: K_ = np.linalg.inv(H).T.dot(circle()).dot(np.linalg.inv(H))
            except: continue
            cX = (v0 - circle()).T.dot(K_).dot(v0 - circle())
            
            E_ = traj.figures(Ellipse, "r" if ox.tag_truth else "b", 1, True)
            traj.ellipse(K_, None, None, E_)
            E_.data.matrix = H
            E_.alpha = 1.0
            
            ox2 = ox
            ox2.hmatrix = H
            break

        #sols = plane_intersection(ox1, ox2)
        #if sols is None: return

        #l1 = traj.figures(Line, "g-", 1, True)
        #l2 = traj.figures(Line, "k-", 1, True)
        #l1.data.intersect = sols["n1"]["sols"]
        #l2.data.intersect = sols["n2"]["sols"]
        #traj.line(sols["n1"]["r0"], sols["n1"]["d"], l1)
        #traj.line(sols["n2"]["r0"], sols["n2"]["d"], l2)

        self.fig.add_object("event" + str(ox.mobius) + str(ox.mobius), traj)
        #self.fig.show()


    def solver(self, ellip, i):
        if len(ellip) == 0: return True
        traj = traject(self.fig)
        ox = ellip
        can = {abs(i.mobius) : i for i in ox}
        if min(can) > 0.1: return True
        sel = max(can)
        ox = can[sel]

        # ---- axis majors ----#
        m_x, m_y, m_z = 1.0, 1.0, -1.0

        # ---- center ----- #
        center = np.array([self.px, self.py, self.pz])
        
        # ---- Radii -----#
        rho = 10

        cov = np.array([
            [m_x,   0,    0], 
            [0,   m_y,    0],
            [0,     0,  m_z]
        ])

        H = ox.hmatrix
        K_ = np.linalg.inv(H).T.dot(cov).dot(np.linalg.inv(H))
        v0 = np.array([[0, 0, self.px],[0, 0, self.py], [0, 0, self.pz]]) *0

        cX = (v0 - cov).T.dot(K_).dot(v0 - cov)
        print("->", cX)
        
        E_ = traj.figures(Ellipse, "r" if ox.tag_truth else "b", 1, True)
        traj.ellipse(cX, None, None, E_)
        E_.data.matrix = cX 
        E_.alpha = 1.0
        
        #c = np.asarray(center).reshape(3)

        #S = np.diag([m_x, m_y, m_z])
        #A = np.linalg.inv(S).dot(np.linalg.inv(S))

        #b_ = -A.dot(c)
        #c_ = c.dot(A.dot(c)) - rho
        #Q   = np.block([
        #        [A          , b_[:, None]     ],
        #        [b_[None, :], np.array([[c_]])]
        #     ])
        #Q   = (Q + Q.T)/2.0
        
        #Q_ = traj.figures(Ellipsoid, "b", 1, True)
        #Q_.data.matrix = Q
        #Q_.alpha = 0.2
  
        self.fig.add_object("event" + str(i), traj)
        return False 
        #self.engine[i].shapes()
        #self.fig.add_object("ellipse-" + str(i+1), self.engine[i])

        return True














