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


def R(axis, angle):
    '''Rotation matrix about x(0),y(1), or z(2) axis'''
    c, s = math.cos(angle), math.sin(angle)
    R = c * np.eye(3)
    for i in [-1, 0, 1]:
        R[(axis-i) % 3, (axis+i) % 3] = i*s + (1 - i*i)
    return R

def Derivative():
    '''Matrix to differentiate [cos(t),sin(t),1]'''
    return R(2, math.pi / 2).dot(np.diag([1, 1, 0]))




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

    @property
    def H_tilde(self):
        '''Transformation of t=[c,s,1] to p_nu: F coord.'''
        x1, y1, p = self.x1, self.y1, self.mu.mag
        Z, w, Om = self.Z, self.w, math.sqrt(self.Om2)
        return np.array([[  Z/Om, 0, x1-p],
                         [w*Z/Om, 0,   y1],
                         [     0, Z,    0]])

    @property
    def H(self):
        '''Transformation of t=[c,s,1] to p_nu: lab coord.'''
        return self.R_T.dot(self.H_tilde)

    @property
    def R_T(self):
        '''Rotation from F coord. to laboratory coord.'''
        b_xyz = self.b.x, self.b.y, self.b.z
        R_z = R(2, -self.mu.phi)
        R_y = R(1, 0.5*math.pi - self.mu.theta)
        R_x = next(R(0,-math.atan2(z,y))
                   for x,y,z in (R_y.dot(R_z.dot(b_xyz)),))
        return R_z.T.dot(R_y.T.dot(R_x.T))

class singleNeutrinoSolution(object):
    '''Most likely neutrino momentum for tt-->lepton+jets'''
    def __init__(self, b, mu, met, sigma2, mW2, mT2):
        metX, metY = met
        self.solutionSet = SolutionSet(b, mu, mW2, mT2, 0)
        S2 = np.vstack([np.vstack([np.linalg.inv(sigma2),
                                   [0, 0]]).T, [0, 0, 0]])
        V0 = np.outer([metX, metY, 0], [0, 0, 1])
        deltaNu = V0 - self.solutionSet.H
        self.X = np.dot(deltaNu.T, S2).dot(deltaNu)
        M = next(XD + XD.T for XD in (self.X.dot(Derivative()),))
        self.V0 = M 
        return 


        solutions = intersections_ellipses(M, UnitCircle())
        self.solutions = sorted(solutions, key=self.calcX2)


