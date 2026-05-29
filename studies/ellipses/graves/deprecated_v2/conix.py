from relations import *
from atomics import *
import math

class NuConuix:
    def __init__(self, lep, bqrk, nu = None, truth = None): 
        self.RT = None
        self.lep = lep
        self.jet = bqrk
        self.cols  = iter(["blue", "navy", "cyan", "green", "blue", "navy", "cyan", "green"])
        self.style = iter(["-.", "-.", "-.", "-.", "-", "-", "-", "-"])

        self.data = variables(lep, bqrk, nu, truth)

    @property 
    def R_T(self):
        if self.RT is not None: return self.RT
        px, py, pz = self.lep.px, self.lep.py, self.lep.pz
        phi   = np.arctan2(py, px)
        theta = np.arctan2(np.sqrt(px**2 + py**2), pz)
        R_z   = rotation_z(-phi)
        R_y   = rotation_y(0.5*np.pi - theta)
        
        b_vec = np.array([self.jet.px, self.jet.py, self.jet.pz])
        b_rot = R_y @ (R_z @ b_vec)
        R_x = rotation_x(-np.arctan2(b_rot[2], b_rot[1]))
        self.RT = R_z.T @ R_y.T @ R_x.T
        return self.RT

