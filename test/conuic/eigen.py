from atomics import *

class eigen:
    def __init__(self): pass

    # NOTE: GXX (normal): beta_mu * cosh(tau) * sin(psi) - Omega * sinh(tau) * cos(psi)
    def GXX(self, tau): return self.lep.b * cosh(tau) * self.spsi - self.o * self.cpsi * sinh(tau) 

    # NOTE: G__ (deriv ): beta_mu * cosh(tau) * cos(psi) - Omega * sinh(tau) * sin(psi)
    def G__(self, tau): return self.lep.b * sinh(tau) * self.spsi - self.o * self.cpsi * cosh(tau)

    # NOTE: Gtx (tanh): Omega * tanh(tau) * cos(psi)  - beta_mu * sin(psi)
    def Gtx(self, tau): return self.o * self.cpsi * tanh(tau) - self.lep.b * self.spsi

    # NOTE: Gt (deriv): Omega * cos(psi) - beta_mu * tanh(tau) * sin(psi)
    def Gt(self, tau):  return self.o * self.cpsi - self.lep.b * tanh(tau) * self.spsi

    # ------------ derivatives --------------- #
    # characteristic polynomial of H_tilde
    def _P(self, l = None, z = None, tau = None): 
        a = -1
        b =  z / self.o
        c = -(z ** 2) / self.o * self.GXX(tau)
        d = -(z ** 3) * sinh(tau) / (self.o * self.cpsi)
        if l is None: return [a, b, c, d]

        o  = a * (l**3)
        o += b * (l**2)
        o += c * (l) 
        o += d 
        return o

    def _dPdL(self, l, z, tau):
        a = -3
        b =  2 / self.o
        c = -1 / self.o
        return a * (l ** 2) + b * (z * l) + c * (z ** 2) * self.GXX(tau)


    def _dPdZ(self, l, z, tau):
        a =  1 / self.o
        b = -2 / self.o
        c = -3 / (self.o * self.cpsi)
        return a * (l ** 2) + b * self.GXX(tau) * z * l + c * (z ** 2) * sinh(tau)

    def _dPdtau(self, l, z, tau):
        a = - 1 /  self.o
        b = - 1 / (self.o * self.cpsi)
        return a * (z ** 2)* l * self.G__(tau) + b * (z**3) * cosh(tau)

    # --------- Special Functions ----------- 
    # NOTE: transfer function -> P(lambda) = 1 / 3 * (lambda * d(P)/d(lambda) + z d(P)/d(z))
    # This is computed when dP/dZ = dP/dLambda = 0.
    def _transfer(self, l, z, t): return 1 / 3 * ( l * self._dPdL(l, z, t) + z * self._dPdZ(l, z, t) )

    # --------- Functions relating to cases where the derivatives are 0.
    # NOTE: This is the value of lambda when dP/dZ = 0.
    def _lambda_dPdZ(self, z, t):
        # dP/dZ = a lambda^2 + 2 b Z lambda - 3 c Z^2
        alpha = self.GXX(t)
        disc = complex(alpha ** 2 + 3 * sinh(t)/self.cpsi)**0.5
        l1, l2 = z * (alpha + disc), z * (alpha - disc)
        return {
                "l1" : l1, "l2" : l2, "discriminant": disc,
                "P(l1)" : self.P(l1, z, t), "P(l2)" : self.P(l2, z, t)
        }

    # NOTE: This is the value of lambda when dP/dlambda = 0.
    def _lambda_dPdL(self, z, t):
        u = tanh(t)
        alpha = 1 / complex(1 - u**2)**0.5
        dx = complex( 1 + 3 * self.o * (self.o * u * self.cpsi - self.lep.b * self.spsi ) * alpha)**0.5
        l1, l2 = (1 + dx) * z/(3 * self.o), (1 - dx) * z/(3 * self.o)
        return {
                "l1" : l1, "l2" : l2, "discriminant" : dx, 
                "P(l1)" : self.P(l1, z, t), "P(l2)" : self.P(l2, z, t)
        }

    # NOTE: This is the value of lambda when dP/dtau = 0.
    def _lambda_dPdtau(self, z, t): return z/(self.Gt(t) * self.cpsi)
 

    # NOTE: This is computes the degenerate roots of dP/dZ = 0
    # This is the case where the discriminant is 0.
    # It does not imply that dP/dZ = 0 and P(lambda) = 0.
    def _lambda_dPdZ_degenerate(self):
        a =  self.o**4 * self.cpsi ** 6 + 9 
        b = -4 * self.o**3 * self.lep.b * self.cpsi ** 5 * self.spsi 
        c =  6 * self.o**2 * self.lep.b ** 2 * self.cpsi ** 4 * self.spsi ** 2 - 9
        d = -4 * self.o * self.lep.b ** 3 * self.spsi ** 3 
        e =  self.lep.b ** 4 * self.spsi ** 4 * self.cpsi ** 2 
        r = np.roots([a, b, c, d, e])
        roots = {}
        for i in r: 
            if abs(i) > 1: continue
            tau = math.atanh(i.real)
            v = self._lambda_dPdZ(1, tau)
            roots[v["discriminant"]] = {"tau" : tau, "roots" : i} | v
        return roots

    # NOTE: This is computes the degenerate roots of dP/dL = 0
    # This is the case where the discriminant is 0.
    # It does not imply that dP/dL = 0 and P(lambda) = 0.
    def _lambda_dPdL_degenerate(self):
        a = 9 * self.o ** 3 * self.lep.b * self.spsi * self.cpsi 
        b = 9 * self.o ** 4 * self.cpsi ** 2 + 1 
        dc = complex(9 * self.o ** 2 * (1 - self.lep.b**2) + 1)**0.5
        u1, u2 = (a + dc)/b, (a - dc)/b

        roots = {}
        for i in [u1, u2]: 
            if abs(i) >= 1: continue
            tau = math.atanh(i.real)
            v = self._lambda_dPdL(1, tau)
            roots[v["discriminant"]] = {"tau" : tau, "roots" : i} | v
        return roots

    # NOTE: Mobius transformation obtained from solving for lambda in dP/dtau = 0
    # then substituting back into P(lambda) and letting P(lambda) = 0
    # The transform is defined:
    # M(tau) = (Omega sin(psi) + beta_mu tanh(tau) cos(psi)) / (Omega cos(psi) - beta_mu u sin(psi))
    # The resulting equation has the form:
    # M(tau)^2 = -(beta_mu cos(psi)^2) (Omega sin(psi) + beta_mu tanh(tau) cos(psi)) sqrt(1 - tanh(tau)^2)
    def _M_transform(self, t):
        u = tanh(t)
        mob  = (self.o * self.spsi + self.lep.b * self.cpsi * u)
        mob /= (self.o * self.cpsi - self.lep.b * self.spsi * u)
        rhs  = -(self.lep.b * self.cpsi**2) * (self.o * self.spsi + self.lep.b * u * self.cpsi) * (1 - u**2)**0.5
        return {"LHS": mob**2, "RHS": rhs}

    def _M_inverse(self, M):
        return atanh(self.o * ( 1 - M * self.w ) / ( self.lep.b * (M + self.w) )) 

    def _M_coef(self):
        a =   self.o     * self.spsi
        b =   self.lep.b * self.cpsi
        c = - self.lep.b * self.spsi
        d =   self.o     * self.cpsi
        tr = self.o * (self.cpsi + self.spsi)
        pole_max = -(self.o * self.w)/self.lep.b
        pole_min = (self.o / (self.lep.b * self.w))
        
        dx = complex( (self.o - self.lep.b)**2 * self.w ** 2 - 4 * self.lep.b * self.o ) ** 0.5
        b = (self.o - self.lep.b) * self.w 
        t1, t2 = (b + dx) / (2 * self.lep.b), (b - dx) / (2 * self.lep.b)
        t1, t2 = atanh(t1), atanh(t2)
        
        M_min = (self.o - self.lep.b * self.w) / (self.o * self.w + self.lep.b)
        M_max = (self.o + self.lep.b * self.w) / (self.o * self.w - self.lep.b)

        if   tr  < 2 * self.w ** 0.5: surf = "elliptic - Riemann rotation"
        elif tr == 2 * self.w ** 0.5: surf = "parabolic - one fixed point"
        elif tr  > 2 * self.w ** 0.5: surf = "hyperbolic - two fixed points"

        return {
                "M-matrix" : [[a, b], [c, d]], 
                "determinant" : self.w, 
                "trace": tr, 
                "M-max" : M_max, "M-min" : M_min,
                "surface" : surf, 
                "pole_max" : pole_max, "pole_min" : pole_min,
                "tau1" : t1, "tau2" : t2
        }
 
    # NOTE: Here we compute the roots of tau where 
    # P = 0 and dPdtau = 0 simultaneously
    def _lambda_roots_dPdtau(self, z, get_all):
        mu2 = (self.lep.mass / self.lep.e)**2
        a = (1 + self.w**2)**3 * self.w ** 2 
        b = 2 * ( 1 + self.w ** 2) ** 3 * self.w 
        c = (1 + self.w**2) ** 3 + self.o ** 2 - self.lep.b ** 2 * self.w ** 2
        d = - 2 * self.w * (self.lep.b ** 2 + self.o ** 2)
        e = self.o**2 * self.w - self.lep.b**2
        mr = np.roots([a, b, c, d, e])
        sols = {}
        for i in mr: 
            tau = self._M_inverse(i)
            if tau is None: continue
            l_ = self._lambda_dPdtau(z, tau)
            p = self._P(l_, z, tau)
            dpdt = self._dPdtau(l_, z, tau)
            null = (abs(dpdt.real)**2 + abs(p.real)**2)**0.5
            sols[null] = {"null": null, "lambda": l_, "tau" : tau, "P": p, "dpdt": dpdt}
        if not len(sols): return sols
        return sols[min(sols)] if get_all is False else sols
    

    # NOTE: Post Analysis for finding Z
    # Here we compute the Z roots from the characteristic polynomial
    def _Z_roots_P(self, l, tau):
        a = - sinh(tau) / (self.o * self.cpsi)
        b = - (l / self.o) * self.GXX(tau) 
        c = l ** 2 / self.o
        d = -l ** 3
        r = np.roots([a, b, c, d])
        r = [k.real.item() for k in r if not abs(k.imag)]
        return [k for k in r if k > 0]
