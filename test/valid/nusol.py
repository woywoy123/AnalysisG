import numpy as np
import vector
import math

def costheta(v1, v2):
    v1_sq = v1.px**2 + v1.py**2 + v1.pz**2
    if v1_sq == 0: return 0

    v2_sq = v2.px**2 + v2.py**2 + v2.pz**2
    if v2_sq == 0: return 0

    v1v2 = v1.px*v2.px + v1.py*v2.py + v1.pz*v2.pz
    return v1v2/math.sqrt(v1_sq * v2_sq)

class Vec:
    """A 4-vector class for kinematics."""
    def __init__(self, px, py, pz, e): self.px, self.py, self.pz, self.e = px, py, pz, e

    @property
    def pvec(self): return np.array([self.px, self.py, self.pz])

    @property
    def mag(self): return np.linalg.norm(self.pvec)

    @property
    def beta(self): return self.mag / self.e

    @property
    def phi(self): return math.atan2(self.py, self.px)

    @property
    def theta(self): return math.acos(self.pz / self.p) 
   
    @property
    def tau2(self):
        return self.e**2 - self.mag**2



class NuSol(object):
    def __init__(self, b, mu, mW2 = 0, mT2 = 0, mN2 = 0):
        self._b = b
        self._mu = mu
        self.mW2 = mW2*mW2
        self.mN2 = mN2*mN2
        self.mT2 = mT2*mT2

    @property
    def b(self): return self._b

    @property
    def mu(self): return self._mu

    @property
    def c(self): return costheta(self._b, self._mu)

    @property
    def s(self): return math.sqrt(1 - self.c**2)

    @property
    def x0p(self):
        m2 = self.b.tau2
        return -(self.mT2 - self.mW2 - m2)/(2*self.b.e)

    @property
    def x0(self):
        m2 = self.mu.tau2
        return -(self.mW2 - m2 - self.mN2)/(2*self.mu.e)

    @property
    def Sx(self):
        P = self.mu.mag
        beta = self.mu.beta
        return (self.x0 * beta - P * (1 - beta**2))/beta**2

    @property
    def Sy(self):
        beta = self.b.beta
        return ((self.x0p / beta) - self.c * self.Sx) / self.s

    @property
    def w(self):
        beta_m, beta_b = self.mu.beta, self.b.beta
        return (beta_m/beta_b - self.c)/self.s

    @property
    def Om2(self): return self.w**2 + 1 - self.mu.beta**2

    @property
    def eps2(self):
        return (self.mW2 - self.mN2) * (1 - self.mu.beta**2)

    @property
    def x1(self):
        return self.Sx - ( self.Sx + self.w * self.Sy)/self.Om2

    @property
    def y1(self):
        return self.Sy - ( self.Sx + self.w * self.Sy)*self.w/self.Om2

    @property
    def Z2(self):
        p1 = (self.x1**2)* self.Om2
        p2 = - (self.Sy - self.w * self.Sx)**2
        p3 = - (self.mW2 - self.x0**2 - self.eps2)
        return  p1 + p2 + p3

    @property
    def Z(self): return math.sqrt(max(0, self.Z2))

    @property
    def BaseMatrix(self):
        return np.array([
            [self.Z/math.sqrt(self.Om2)           , 0   , self.x1 - self.mu.mag],
            [self.w * self.Z / math.sqrt(self.Om2), 0   , self.y1              ],
            [0,                                   self.Z, 0                    ]])


if __name__ == "__main__":

    muons = [
        [14.979, 42.693,  -79.570,  91.534],
        [60.236, 69.537,  166.586, 190.302],
        [20.575, 27.615,  100.242, 294.501]
    ]

    bjets = [
        [-23.487, 116.748, -64.443, 136.770],
        [114.379, -48.805, 167.815, 209.192],
        [ 19.069, -58.705, -10.629,  62.940]
    ]


    mT = 172.68 
    mW = 80.385
    nu = NuSol(Vec(*bjets[0]), Vec(*muons[2]), mW, mT)
    print("nu.c   ", nu.c    ) 
    print("nu.s   ", nu.s    )
    print("nu.x0p ", nu.x0p  )
    print("nu.x0  ", nu.x0   )
    print("nu.Sx  ", nu.Sx   )
    print("nu.Sy  ", nu.Sy   )
    print("nu.w   ", nu.w    )
    print("nu.Om2 ", nu.Om2  )
    print("nu.eps2", nu.eps2 )
    print("nu.x1  ", nu.x1   )
    print("nu.y1  ", nu.y1   )
    print("nu.Z2  ", nu.Z2   )
    print(nu.BaseMatrix)
    print(nu.x1 - nu.mu.mag)

#nu.Z2   19946.44729279096







#nu.c      0.7874843844071331
#nu.s      0.6163346041842196

#nu.x0p  -84.02899085691305
#x0p()   -84.028990856913054586

#nu.x0   -35.29680885052674
#x0()    -35.296808850526744550

#nu.Sx   -35.29713415087069
#Sx()    -35.297134150870697056

#nu.Sy   -92.61159276753646
#Sy()    -92.611592767536464521

#nu.w      0.36115149345418496

#nu.Om2    0.13043338063761978

#nu.eps2   0.01925221945341718

#nu.x1   491.7454745278624
#x1()    491.745474527862427294

#nu.y1    97.73063247077759
#y1()     97.730632470777536014


#Z2()     26407.176268550923850853

