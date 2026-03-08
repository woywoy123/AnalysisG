import sympy as sp
from helper import *

class verified:
    def __init__(self, bq = None, mu = None, nu = None, theta = None):
        if bq is None: bq = particle("b")
        if mu is None: mu = particle("mu")
        if nu is None: nu = particle("nu")
        if theta is None: theta = symbol("theta")
        self.kp = mu.beta / bq.beta + 1 #symbol("k+")
        self.km = mu.beta / bq.beta - 1 #symbol("k-")
        self.psi = symbol("psi")
        
        self.bq = bq
        self.mu = mu
        self.nu = nu
        self.theta = theta
        self.cth = sp.cos(theta)
        self.sth = sp.sin(theta)

    def mW2(self, Sx): 
        return - 2 * self.mu.p * Sx - self.mu.mass ** 2 + self.nu.mass ** 2 

    def mT2(self, Sx, Sy):
        clm =     - 2 * Sx * (self.mu.p + self.bq.p * self.cth) 
        clm = clm - 2 * self.bq.p * Sy * self.sth
        clm = clm - self.mu.mass ** 2 + self.nu.mass ** 2 + self.bq.mass ** 2 
        return clm 

    def Z2(self, Sx, Sy, w, O = None):
        if O is None: O = self.O2(w)
        else: O = O ** 2
        b_mu, m_mu, p_mu, m_nu = self.mu.beta, self.mu.mass, self.mu.p, self.nu.mass
        A = (b_mu ** 2 - w ** 2) / O
        B = 2 * w / O
        C = - (1 - b_mu**2)/O
        D = 2 * p_mu 
        E = m_mu ** 2 - m_nu ** 2
        return A * Sx ** 2 + B * Sx * Sy + C * Sy ** 2 + D * Sx + E 

    def O2(self, w): return w ** 2 + 1 - self.mu.beta ** 2
    def O(self, w):  return sp.sqrt(self.O2(w))
  
    @property
    def wpk(self): return 0.5 * (self.kp * sp.tan(self.psi) + self.km / sp.tan(self.psi))

    @property
    def wmk(self): return -0.5 * (self.kp * 1 / sp.tan(self.psi) + self.km * sp.tan(self.psi))

    @property
    def _wm(self): return - (1 / self.sth) * ( self.mu.beta / self.bq.beta + self.cth)

    @property
    def _wp(self): return   (1 / self.sth) * ( self.mu.beta / self.bq.beta - self.cth)


class dG2(verified):
    def __init__(self):
        verified.__init__(self)
        self.wp = symbol("w_{+}")
        self.op = symbol("O_{+}")

        self.wm = symbol("w_{-}")
        self.om = symbol("O_{-}")

        self.Sx = symbol("Sx")
        self.Sy = symbol("Sy")

        self.dp = symbol("delta^+")
        self.dm = symbol("delta^-")

    @property
    def GammaP(self): return (self.wp + self.wm) / self.O2(self.wp)

    @property
    def GammaM(self): return (self.wp - self.wm) / self.O2(self.wm)

    @property
    def deltaP(self): 
        op, om = self.O(self.wp), self.O(self.wm)
        return ( (op - om) ** 2 - (self.wp + self.wm) ** 2 ) / (2 * (self.wp + self.wm))

    @property
    def deltaM(self): 
        op, om = self.O(self.wp), self.O(self.wm)
        return ( (op + om) ** 2 - (self.wp + self.wm) ** 2 ) / (2 * (self.wp + self.wm))
    
    @property
    def kappaM(self): return 1 / self.deltaM

    @property
    def kappaP(self): return 1 / self.deltaP
    
    @property
    def lineP(self): 
        dp = ( (self.op - self.om) ** 2 - (self.wpk + self.wmk) ** 2 ) / (2 * (self.wpk + self.wmk))
        return self.Sx * dp

    @property
    def lineM(self): 
        dp = ( (self.op + self.om) ** 2 - (self.wpk + self.wmk) ** 2 ) / (2 * (self.wpk + self.wmk))
        return self.Sx * dp

    def factored(self, Sx, Sy):
        F = -self.GammaP * self.GammaM 
        return F * (Sx - self.deltaP * Sy) * (Sx - self.deltaM * Sy)
    



