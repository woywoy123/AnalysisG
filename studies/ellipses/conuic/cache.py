from atomics import *
from mobius import *
import numpy as np
import cmath 

class matrix:
    def __init__(self):
        self.RT = None

    def P(self, z, l, tau):
        a = - l**3
        b = (z/self.o) * l**2
        c = - (z**2) * (l / self.o) * (self._b * self.spsi * cosh(tau) - self.o * self.cpsi * sinh(tau))
        d = - (z**3) * (1 / (self.o * self.cpsi)) * sinh(tau)
        return a + b + c + d

    def dPdt(self, z, l, tau):
        a = - (l * z ** 2) * (1 / self.o) * (self._b * self.spsi * sinh(tau) - self.o * self.cpsi * cosh(tau))
        b = - (z ** 3) * cosh(tau) / (self.o * self.cpsi)
        return a + b

    def dP2d2t(self, z, l, tau):
        a = -  (z**2) * (l / self.o) * (self._b * self.spsi * cosh(tau) - self.o * self.cpsi * sinh(tau))
        b = - ((z**3) / (self.o * self.cpsi)) * sinh(tau)
        return a + b

    def cache(self):
        self.cos = costheta(self.lep, self.jet)
        self.sin = (1 - self.cos**2) ** 0.5
        self._b = self.lep.b
        self._m = self.lep.mass
        self._e = self.lep.e
        
        self.w   = (self.lep.b/self.jet.b - self.cos)/self.sin
        self.o2  = self.w**2 + 1 - self.lep.b ** 2
        self.o   = self.o2 ** 0.5

        # ------- psi declare ----------#
        self.tpsi = self.w
        self.cpsi = 1.0 / (1 + self.w**2)**0.5
        self.spsi = self.tpsi * self.cpsi
        rt = self.R_T

        # ------- HBAR --------- #
        self.HB1 = nulls(3,3)
        self.HB1[0][0] = 1
        self.HB1[1][0] = self.tpsi 
        self.HB1[2][1] = self.o
        self.HB1 = np.array(self.HB1)
        self.HT1 = self.HB1.dot(rt)
      
        self.HB2 = nulls(3,3)
        self.HB2[0][2] = -self._b * self.cpsi
        self.HB2[1][2] = -self._b * self.spsi
        self.HB2 = np.array(self.HB2)
        self.HT2 = self.HB2.dot(rt)
     
        self.HB3 = nulls(3,3)
        self.HB3[0][2] = -self.o * self.spsi
        self.HB3[1][2] =  self.o * self.cpsi
        self.HB3 = np.array(self.HB3)
        self.HT3 = self.HB3.dot(rt)

        # ------ poles
        self.pn =  (self.o / self._b) * (1 / self.tpsi) # Asymptote
        self.pm = -(self.o / self._b) * self.tpsi       # Z / Omega solution

        self.mob = mobius(self)

    def test(self, tau):

        ## ------- Asympotitic solutions --------#
        #assert self.P(1, 0, 1) - self.dP2d2t(1, 0, 1) == 0 # lambda = 0
        #assert round((self.P(1, (1 / self.o), 1) - self.dP2d2t(1, (1 / self.o), 1)), 8) == 0 # lambda = Z / Omega
        #
        ## ------ tau solutions --------#
        ## ----- First the denominator of dPdtau has to be zero -> inf
        #assert round(self._b * self.spsi * self.pn - self.o * self.cpsi, 8) == 0
      
        # ----- Second the lambda value or tau have to be zero or non-real
        #tau = atanh(self.pm)
        #if tau is not None: assert round(self.P(1, self.dPdtL0(1, tau), tau) - self.dP2d2t(1, self.dPdtL0(1, tau), tau), 8) == 0

        import numpy 
        self.mob.u = numpy.array([self.mob.pole_P()])
        for i in range(1000):
            un = self.mob.u 
            self.mob.u = self.mob.newton()
            if abs(un - self.mob.u) == 0: break
            if abs(self.mob.condition()) > 10**-10: continue
            break

        self.tau   = cmath.atanh(self.mob.u)
        self.error = self.mob.condition()
        if abs(self.error) > 10**-10 or numpy.isnan(self.error[0]): self.error = None; return 

        u1p, u1m = self.mob.alpha_pm()
        u2p, u2m = self.mob.alpha_mp()

        self.l = self.dPdtL0(1, tau)
        self.phi = numpy.tan(self.mob.phi())
        # got no idea why this works!
        print(self.error, self.mob.phi(), self.l, self.is_truth)
        if abs(self.mob.phi()) < 3: self.error = None; return 
        print("____passed----")
        self.nu = self.Hmatrix(1, self.tau).dot([math.cos(self.phi), math.sin(self.phi), 1])
        print(u1p, u1m, u2p, u2m)
        # --------- u'-----
        #ux     = self.mob.uprime() 
        #print(self.mob.kfactor(None, ux))
        #print("++:", (1 - u1p)*(1 + u2p), self.mob.test_sol1(u1p, u2p, ux))
        #print("+-:", (1 - u1p)*(1 + u2m), self.mob.test_sol1(u1p, u2m, ux))
        #print("-+:", (1 - u1m)*(1 + u2p), self.mob.test_sol1(u1m, u2p, ux))
        #print("--:", (1 - u1m)*(1 + u2m), self.mob.test_sol1(u1m, u2m, ux))
        print("->", self.mob.phi(), self.masses(1, self.tau))
        #print(self.nu)
        #print(self.error)
        #print(self.dP2d2t(1, self.l, self.tau))
        #print("__________")















