import cmath
import numpy
import math

class mobius:

    def __init__(self, obj):
        self.Omega = obj.o
        self.beta  = obj._b
        self.cpsi  = obj.cpsi
        self.spsi  = obj.spsi
        self.tpsi  = obj.tpsi

    def condition(self): return self.beta * self.cpsi ** 2 * (1 - self.u**2)**0.5 + self.aplx() / self.aplm()**2

    def pole_P(self): return - (self.Omega / self.beta) * self.tpsi 
    def pole_M(self): return   (self.Omega / self.beta) * 1.0 / self.tpsi

    def aplx(self): return self.Omega * self.spsi + self.beta * self.cpsi * self.u
    def aplm(self): return self.Omega * self.cpsi - self.beta * self.spsi * self.u

    # Condition where alpha^+/alpha^- = 1 - u
    def alpha_pm(self): 
        D  = (self.beta * (self.spsi + self.cpsi) + self.Omega * self.cpsi)**2
        D += 4 * self.beta * self.Omega * (self.spsi - self.cpsi) * self.spsi
        D = complex(D)**0.5
        r1 = (self.beta * (self.spsi + self.cpsi) + self.Omega * self.cpsi + D)/(2 * self.beta * self.spsi)
        r2 = (self.beta * (self.spsi + self.cpsi) + self.Omega * self.cpsi - D)/(2 * self.beta * self.spsi)
        return r1, r2


    # Condition where alpha^-/alpha^+ = 1 + u
    def alpha_mp(self): 
        D  = (self.beta * (self.spsi + self.cpsi) + self.Omega * self.spsi)**2
        D -= 4 * self.beta * self.Omega * (self.spsi - self.cpsi) * self.cpsi
        D = complex(D)**0.5
        r1 = (-self.beta * (self.spsi + self.cpsi) - self.Omega * self.spsi + D)/(2 * self.beta * self.cpsi)
        r2 = (-self.beta * (self.spsi + self.cpsi) - self.Omega * self.spsi - D)/(2 * self.beta * self.cpsi)
        return r1, r2

    # beta_mu cos^2(psi) + (1 - u')/alpha^-(u') = 0
    def uprime(self):
        u = 1 + self.beta * self.Omega * self.cpsi**3
        d = 1 + self.beta ** 2 * self.spsi * self.cpsi ** 2
        return u / d
    
    def kfactor(self, u = None, ux = None):
        u1p, u1m = self.alpha_pm()
        u2p, u2m = self.alpha_mp()

        kp = (u2p - u1m)/(u2p - u1p)
        km = (u2m - u1m)/(u2m - u1p)
        
        if u  is not None: return  kp * (u - u1p)/(u - u1m), km * (u - u1p)/(u - u1m)
        if ux is not None: return (kp * u1p - ux *u1m)/(kp - ux), (km * u1p - ux *u1m)/(km - ux)
        return kp, km 

    def fixed_points(self):
        dc = self.cpsi * complex( (self.beta - self.Omega) ** 2 - 4 * self.beta * self.Omega * self.tpsi ** 2)**0.5
        u_p = - self.cpsi * ( self.beta - self.Omega ) + dc
        u_m = - self.cpsi * ( self.beta - self.Omega ) - dc
        return u_p / (2 * self.beta * self.spsi), u_m / (2 * self.beta * self.spsi)

    def normal_form(self):
        u_p, u_m = self.fixed_points()
        k = (self.tpsi - u_p)/(self.tpsi + u_m) * (u_m / u_p)
        return k * (self.u - u_p)/(self.u - u_m)

    def eigenvalues(self):
        dc = complex( self.cpsi ** 2 * (self.beta + self.Omega) ** 2 - 4 * self.beta * self.Omega )**0.5  
        u_p = self.cpsi * ( self.beta + self.Omega ) + dc
        u_m = self.cpsi * ( self.beta + self.Omega ) - dc
        return u_p / 2, u_m / 2

    def mobius_Matrix(self):
        return numpy.array([
           [ self.beta * self.cpsi, self.Omega * self.spsi],
           [-self.beta * self.spsi, self.Omega * self.cpsi]
        ])
    
    def phi(self):
        a = complex(4 * self.beta * self.Omega - self.cpsi ** 2 * (self.beta + self.Omega) ** 2) ** 0.5
        a = a / (self.cpsi * (self.beta + self.Omega)) 
        return 2 * numpy.atan2(a.imag, a.real)

    def newton(self):
        dx = - (self.beta * self.u * self.cpsi ** 2) / (1 - self.u ** 2) ** 2
        dy = self.beta * (self.aplm() * self.cpsi + 2 * self.aplx() * self.spsi) / self.aplm() ** 3
        
        x = self.beta * self.cpsi ** 2 * (1 - self.u ** 2) ** 0.5 + self.aplx() / self.aplm() ** 2
        return self.u - x / (dx + dy)

    def input(self, tau): self.u = numpy.tanh(tau)

    def test_sol1(self, kp, km, uv):
        fx = (1 - uv) / (self.Omega * self.cpsi - self.beta * self.spsi * uv)
        return self.beta * self.cpsi ** 2 * complex((1 - kp) * (1 + km))**0.5 + fx




