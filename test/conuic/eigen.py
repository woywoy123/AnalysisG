from atomics import *

class eigen:
    def __init__(self): pass

    # characteristic polynomial of H_tilde
    def _P(self, l = None, z = None, tau = None): 
        a =  1
        b = -z / self.o
        c =  (z ** 2) / self.o * self.GXX(tau)
        d =  (z ** 3) * sinh(tau) / (self.o * self.cpsi)
        if l is None: return [a, b, c, d]

        o  = a * (l**3)
        o += b * (l**2)
        o += c * (l) 
        o += d 
        return o

    def _dPdL(self, l, z, tau):
        a =  3
        b = -2 / self.o
        c =  1 / self.o
        return a * (l ** 2) + b * (z * l) + c * (z ** 2) * self.GXX(tau)


    def _dPdZ(self, l, z, tau):
        a = - 1 / self.o
        b =   2 / self.o
        c =   3 / (self.o * self.cpsi)
        return a * (l ** 2) + b * self.GXX(tau) * z * l + c * (z ** 2) * sinh(tau)

    def _dPdtau(self, l, z, tau):
        a = 1 /  self.o
        b = 1 / (self.o * self.cpsi)
        return a * (z ** 2)* l * self.G__(tau) + b * (z**3) * cosh(tau)

    # NOTE: --------- check for special roots ---------- 

    # NOTE: transfer function -> P(lambda) = 1 / 3 * (lambda * d(P)/d(lambda) + z d(P)/d(z))
    # This is computed when dP/dZ = dP/dLambda = 0.
    def _transfer(self, l, z, t): return 1 / 3 * ( l * self._dPdL(l, z, t) + z * self._dPdZ(l, z, t) )

    # NOTE: This is the value of lambda when dP/dtau = 0.
    def _lambda_dPdtau(self, z, t): return z/(self.Gt(t) * self.cpsi)

    # NOTE: ------- G factors ---- consequence of dP/dtau = lambda -> P(..) = 0
    # This was used to generate the _roots_tau function:
    # Much simpler definition. 
    # cos(psi)^2 alpha^2 sqrt( (G^- + alpha) (G^+ - alpha) ) - (alpha - Omega) =  0
    def _lambda_tau_P(self, t):
        alpha = self.Gt(t) * self.cpsi
        a = self.lep.b * self.spsi * self.cpsi
        b = self.o * self.cpsi - alpha
        gamma = (a - b) * (a + b)
        return self.cpsi * ( self.o - alpha ) * gamma ** 0.5 + alpha ** 2 * gamma

    # NOTE: --------- a numerically easier way to test if real ----------- #
    # ----> if this returns a real number close to zero then 
    # ----> dP/dt = 0 and P = 0 @ t . 
    # ----> ALSO HAS NO Z DEPENDENCY!!!
    def _roots_tau(self, t):
        alpha = self.Gt(t) * self.cpsi
        b1 = complex(self.o / self.spsi) ** 0.5 - alpha * self.spsi * complex(self.lep.b * cosh(t)) ** 0.5
        b2 = complex(self.o / self.spsi) ** 0.5 + alpha * self.spsi * complex(self.lep.b * cosh(t)) ** 0.5

        b3  = complex(alpha * ( self.cpsi - alpha * self.lep.b * self.spsi * cosh(t) )) ** 0.5
        b3 -= alpha * self.cpsi * complex(sinh(t) * ( alpha  - self.o * self.cpsi    )) ** 0.5

        b4  = complex(alpha * ( self.cpsi - alpha * self.lep.b * cosh(t) * self.spsi )) ** 0.5
        b4 += alpha * self.cpsi * complex(sinh(t) * ( alpha  - self.o * self.cpsi    )) ** 0.5

        # NOTE: this is a root only if sin(psi) B1 B2 - B3 B4 = 0
        disc = self.spsi * (b1 * b2) / (b3 * b4) - 1 
        l = self._lambda_dPdtau(1, t)
        return {
                "real"   : disc.real, "imag" : disc.imag, 
                "lambda" : self._lambda_dPdtau(1, t), "tau" : t,
                "dPdtau" : self._dPdtau(l, 1, t), 
                "dPdZ"   : self._dPdZ(l, 1, t), 
                "dPdL"   : self._dPdL(l, 1, t),
                "P"      : self._P(l, 1, t),
                "b1" : b1, "b2" : b2, "b3" : b3, "b4" : b4
        }


    def _mobius(self):
        r = (1 + self.w ** 2) ** 0.5
        s = (1 - self.lep.b ** 2)
        
        a = 1
        b = - 2 * self.o 
        c = - r ** 2 * s
        d =   self.lep.b * self.w * r ** 3 
        e = - self.lep.b * self.w * self.o * r ** 5

        x = np.roots([a, b, c, d, e])
        x = [np.atanh((self.o - i.real)/(self.lep.b * self.w)) for i in x if not abs(i.imag)]
        x = [(t, self._lambda_dPdtau(1, t)) for t in x]
        x = [(t.item(), l, self._dPdtau(l, 1, t), self._P(l, 1, t)) for (t, l) in x if not math.isnan(t)]
        print(x)


        #g = self.o - self.lep.b * self.w * tanh(t)









    def _test(self, l, z, t):
        truth = self.P(l, z, t)
        G, Gt, bmu, cpsi, spsi, o = self.GXX(t), self.Gtx(t), self.lep.b, self.cpsi, self.spsi, self.o

        x = l/z
        tx = (1 / 3) * (1/z) * (x * self._dPdL(l, z, t) + self._dPdZ(l, z, t))
        #print(self._transfer(l, z, t), truth)
      
        # ----- roots 
        x = (1 + (1 - 3 * o * G) ** 0.5)/(3 * o)
        dl = 3 * o * x ** 2 - 2 * x + G 

        x = G + ( G**2 + 3 * sinh(t)/cpsi) ** 0.5
        dz = (- x ** 2 + 2 * G * x + 3 * sinh(t) / cpsi)
        #print(dz, dl)
        #print(o, cpsi, bmu)

    
        # ----------- degenerate root ------- # 
        deg = math.asinh(- cpsi / (27 * o ** 2))
        x = 1 / (3 * o)
        G = 1 / (3 * o)
        sx = -cpsi / (27 * o ** 2)

        dl = 3 * o * x ** 2 - 2 * x + G 
        dz = (- x ** 2 + 2 * G * x + 3 * sx / cpsi)
        #print(dz, dl)

        l = z / (3 * o)

        x = (1 + (1 - 3 * o * self.GXX(deg)) ** 0.5)/(3 * o)
        dl = 3 * o * x ** 2 - 2 * x + G 

        x = self.GXX(deg) + complex( self.GXX(deg)**2 + 3 * sinh(deg) / cpsi )**0.5
        dz = (- x ** 2 + 2 * self.GXX(deg) * x + 3 * sinh(deg) / cpsi)

        if dz.imag == 0: return
        print(deg)
        print("->", dl, dz.real, ">" if dz.imag == 0 else "<-------------")
        print("+> ", self._P(l, z , deg), self._dPdL(l, z, deg), self._dPdZ(l, z, deg))
        


    # NOTE: GXX (normal): beta_mu * cosh(tau) * sin(psi) - Omega * sinh(tau) * cos(psi)
    def GXX(self, tau): 
        return self.lep.b * cosh(tau) * self.spsi - self.o * self.cpsi * sinh(tau) 

    # NOTE: G__ (deriv ): beta_mu * cosh(tau) * cos(psi) - Omega * sinh(tau) * sin(psi)
    def G__(self, tau): 
        return self.lep.b * sinh(tau) * self.spsi - self.o * self.cpsi * cosh(tau)

    # NOTE: Gtx (tanh): Omega * tanh(tau) * cos(psi)  - beta_mu * sin(psi)
    def Gtx(self, tau):
        return self.o * self.cpsi * tanh(tau) - self.lep.b * self.spsi

    # NOTE: Gt (deriv): Omega * cos(psi) - beta_mu * tanh(tau) * sin(psi)
    def Gt(self, tau):
        return self.o * self.cpsi - self.lep.b * tanh(tau) * self.spsi



    # --------- check for special roots ---------- #
    # NOTE: Check for dPdZ = 0 -> solve for Lambda
    def _dPdZ_L(self, z, tau):
        G    = complex(self.GXX(tau), 0)
        disc = complex( G ** 2 + 3 * sinh(tau) / self.cpsi ) ** 0.5
        return { "l0" : z * (-G + disc), "l1" : z * (-G - disc) }


    # NOTE: Check for dPdZ = 0 -> find where Lambda is degenerate
    def _dPdZ_D(self, z = None, tau = None):
        # dP/dZ = a lambda^2 + 2 b Z lambda - 3 c Z^2
        # ->  lambda = Z(-G + sqrt( G^2 + 3 sinh(tau) / cos(psi) ) ) 
        # -> Degenerate if sqrt(G^2 + 3 sinh(t) / cos(psi)) = 0.
        # -> Lambda @ Degenerate: Lamba = -Z ( G ) 
        # -> But we need to find tau where sqrt(..) = 0.

        # --- After rewriting sqrt(...) in terms of t = tanh(tau) we get:
        # Omega^2 cos^2(psi) ( t cos(psi) - beta_mu/Omega sin(psi) )^4 - 9 t^2 ( 1 - t^2 ) = 0.
        f = 2 * self.o * self.lep.b * self.tpsi - 3 / ( self.cpsi ** 3 ) 
        s = 3 / ( self.cpsi ** 3 ) * ( 3 / ( self.cpsi ** 3 ) - 4 * self.o * self.lep.b * self.tpsi)

        t1 = (complex(f) + complex(s) ** 0.5) / (2 * self.o2)

        tx = (self.o * t1 - self.lep.b * ( self.o2 + self.lep.b**2 - 1 ) ** 0.5 ) ** 4 
        ty = - 9 * (self.o ** 2 + self.lep.b ** 2) ** 3 
        ty *= t1 ** 2 
        ty *= (1 - t1 ** 2)
        return tx + ty






