from atomics import *

class eigen:
    def __init__(self): pass

    # characteristic polynomial of H_tilde
    def P(self, L, Z, tau): 
        a = -1
        b =  1 / self.o
        c =  1 / self.o
        d = -1 / (self.o * self.cpsi)

        o  = a *          (L**3)
        o += b *  Z     * (L**2)
        o += c * (Z**2) *  L * self.GXX(tau) 
        o += d * (Z**3) *          sinh(tau) 
        return o

    def dPdL(self, L, Z, tau):
        a = -3
        b =  2 / self.o
        c =  1 / self.o
        return a * (L**2) + b * (Z * L) + c * (Z**2) * self.GXX(tau)


    def dPdZ(self, L, Z, tau):
        a =   1 / self.o
        b =   2 / self.o
        c = - 3 / (self.o * self.cpsi)
        return a * (L**2) + b * self.GXX(tau) * Z * L + c * (Z**2) * sinh(tau)

    def dPdtau(self, L, Z, tau):
        a =  1 /  self.o
        b = -1 / (self.o * self.cpsi)
        return (Z ** 2)* L * self.G__(tau) + b * (Z**3) * cosh(tau)

    # --------- check for special roots ---------- #
    # NOTE: Check for dPdZ = 0 -> solve for Z
    # NOTE: 





    # NOTE: GXX (normal): Omega * sinh(tau) * cos(psi) - beta_mu * cosh(tau) * sin(psi)
    def GXX(self, tau): 
        return self.o * self.cpsi * sinh(tau) - self.lep.b * cosh(tau) * self.spsi

    # NOTE: G__ (deriv ): Omega * sinh(tau) * cos(psi) - beta_mu * cosh(tau) * sin(psi)
    def G__(self, tau): 
        return self.o * self.cpsi * cosh(tau) - self.lep.b * sinh(tau) * self.spsi
