import numpy as np
import math
try: from scipy.optimize import leastsq
except: leastsq = None

def CosTheta(v1, v2):
    '''Function to replace ROOT.Math.VectorUtils.CosTheta()'''
    v1_sq = v1.x**2 + v1.y**2 + v1.z**2
    v2_sq = v2.x**2 + v2.y**2 + v2.z**2
    if v1_sq == 0 or v2_sq == 0:
        return 0
    v1v2 = v1.x*v2.x + v1.y*v2.y + v1.z*v2.z
    return v1v2/(v1_sq*v2_sq)**0.5


class SolutionSet(object):
    '''Definitions for nu analytic solution, t->b,mu,nu'''

    def __init__(self, b, mu, mW2, mT2, mN2):
        c = CosTheta(b, mu)
        s = math.sqrt(1-c**2)

        x0p = - (mT2 - mW2 - b.tau2) / (2*b.e)
        x0 = - (mW2 - mu.tau2 - mN2) / (2*mu.e)

        Bb, Bm = b.beta, mu.beta
        
        Sx = (x0 * Bm - mu.mag*(1-Bm**2)) / Bm**2
        Sy = (x0p / Bb - c * Sx) / s

        w = (Bm / Bb - c) / s
        w_ = (-Bm / Bb - c) / s

        Om2 = w**2 + 1 - Bm**2
        eps2 = (mW2 - mN2) * (1 - Bm**2)
        x1 = Sx - (Sx+w*Sy) / Om2
        y1 = Sy - (Sx+w*Sy) * w / Om2
        Z2 = x1**2 * Om2 - (Sy-w*Sx)**2 - (mW2-x0**2-eps2)
        Z = math.sqrt(max(0, Z2))

        for item in ['b','mu','c','s','x0','x0p',
                     'Sx','Sy','w','w_','x1','y1',
                     'Z','Om2','eps2','mW2']:
            setattr(self, item, eval(item))
