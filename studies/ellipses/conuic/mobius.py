from atomics import *
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

    def alpha_p(self, x): return self.Omega * self.tpsi + self.beta * x
    def alpha_m(self, x): return self.Omega - self.tpsi * self.beta * x

    def matrix(self): 
        return np.array([
            [-self.beta * self.tpsi, self.Omega], 
            [self.beta , self.Omega * self.tpsi]
        ])

    def eigenval(self):
        f = self.tpsi * (self.Omega - self.beta) 
        c = complex(self.tpsi ** 2 * (self.Omega - self.beta)**2 + 4 * self.Omega * self.beta * (1 + self.tpsi**2)) ** 0.5
        return (f + c)/2, (f-c)/2

    def eigenvec(self):
        v1, v2 = self.eigenval()
        return np.array([[1, (self.beta * self.tpsi + v1) / self.Omega ], [1, (self.beta * self.tpsi + v2) / self.Omega ]])

    def kfactor(self):
        l1 = (self.Omega - self.beta) * self.tpsi
        l2 = 4 * self.beta * self.Omega * (1 + self.tpsi**2)
        return (l1 - complex(l1 ** 2 + l2))/(l1 + complex(l1 ** 2 + l2))

    def fixed_points(self):
        c = complex((self.Omega * self.tpsi + self.beta * self.tpsi) ** 2 + 4 * self.beta * self.Omega)**0.5
        f = - (self.Omega + self.beta) * self.tpsi
        return (f + c)/(2 * self.beta), (f - c)/(2 * self.beta)
   
    def midpoint(self): return - self.tpsi * ( self.Omega + self.beta ) / (2 * self.beta)

    def dPl0(self, x, use_tanh = False):
        x = np.tanh(x) if use_tanh else x
        ap, am = self.alpha_p(x), self.alpha_m(x)
        return self.beta * self.cpsi ** 3 * (1 - x**2) ** 0.5 * am - ap / am

    def newton_method(self, tol=1e-10, max_iter=100):
        def f(x): 
            ap = self.alpha_p(x)
            am = self.alpha_m(x)
            return (self.beta * np.sqrt(1 - x**2) * am**2 - (1 / self.cpsi) ** 3 * ap)

        def f_prime(x):
            am = self.alpha_m(x)
            sqx = np.sqrt(1 - x**2)

            d_quad = -2 * am * self.beta * self.tpsi
            return  - self.beta * (x / sqx) * am **2 + self.beta * sqx * d_quad - (1 / self.cpsi) ** 3 * self.beta

        x = self.midpoint()
        for i in range(max_iter):
            fx = f(x)
            fpx = f_prime(x)
            print("->", x, fx, fpx)
            if abs(fpx) < 1e-15: return x
            delta = fx / fpx
            x = x - delta
            if abs(delta) > tol: continue
            return x
        return x

