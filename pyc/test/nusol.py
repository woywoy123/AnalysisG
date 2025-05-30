import vector
import math
import numpy as np
from scipy.optimize import leastsq
mW = 80.385*1000
mT = 172.0*1000
mN = 0

def costheta(v1, v2):
    v1_sq = v1.x**2 + v1.y**2 + v1.z**2
    if v1_sq == 0: return 0

    v2_sq = v2.x**2 + v2.y**2 + v2.z**2
    if v2_sq == 0: return 0

    v1v2 = v1.x*v2.x + v1.y*v2.y + v1.z*v2.z
    return v1v2/math.sqrt(v1_sq * v2_sq)

def UnitCircle(): return np.diag([1, 1, -1])

def R(axis, angle):
    """Rotation matrix about x(0),y(1), or z(2) axis"""
    c, s = math.cos(angle), math.sin(angle)
    R = c * np.eye(3)
    for i in [-1, 0, 1]:
        R[(axis - i) % 3, (axis + i) % 3] = i * s + (1 - i * i)
    return R

def Derivative():
    """Matrix to differentiate [cos(t),sin(t),1]"""
    cx = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    return cx.dot(np.diag([1, 1, 0]))

def multisqrt(y):
    """Valid real solutions to y=x*x"""
    return [] if y < 0 else [0] if y == 0 else (lambda r: [-r, r])(math.sqrt(y))


# A = ellipse, Q[i] = line
def intersections_ellipse_line(ellipse, line, zero=1e-10):
    """Points of intersection between ellipse and line"""
    _, V = np.linalg.eig(np.cross(line, ellipse).T)
    sols = sorted([(
        v.real / v[2].real,
        math.log10(sum((line*v.real).tolist())**2 + sum((np.dot(v.real, ellipse) * v.real).tolist())**2)
        ) for v in V.T if v[2].real != 0 and not sum(v.imag)], key = lambda k : k[1])
    return [s for s, k in sols[:2] if k < np.log10(zero)]

def cofactor(A, i, j):
    """Cofactor[i,j] of 3x3 matrix A"""
    a = A[
        not i : 2 if i == 2 else None : 2 if i == 1 else 1,
        not j : 2 if j == 2 else None : 2 if j == 1 else 1,
    ]
    return (-1) ** (i + j) * (a[0, 0] * a[1, 1] - a[1, 0] * a[0, 1])


def factor_degenerate(G, zero=0):
    """Linear factors of degenerate quadratic polynomial"""
    if G[0, 0] == 0 == G[1, 1]: return [[G[0, 1], 0, G[1, 2]], [0, G[0, 1], G[0, 2] - G[1, 2]]]
    swapXY = abs(G[0, 0]) > abs(G[1, 1])
    Q = G[(1, 0, 2),][:, (1, 0, 2)] if swapXY else G
    Q /= Q[1, 1]

    q22 = cofactor(Q, 2, 2)
    if -q22 <= zero: lines = [[Q[0, 1], Q[1, 1], (Q[1, 2] + s)] for s in multisqrt(-cofactor(Q, 0, 0))]
    else:
        x0, y0 = [cofactor(Q, i, 2) / q22 for i in [0, 1]]
        lines = [ [m, Q[1, 1], (-Q[1, 1] * y0 - m * x0)] for m in [Q[0, 1] + s for s in multisqrt(-q22)] ]

    return [[L[swapXY], L[not swapXY], L[2]] for L in lines]

def intersections_ellipses(A, B, returnLines=False, zero = 10e-10):
    """Points of intersection between two ellipses"""
    LA = np.linalg
    sw = abs(LA.det(B)) > abs(LA.det(A))
    if sw: A, B = B, A
    t = LA.inv(A).dot(B)
    e = next(iter([e.real for e in LA.eigvals(t) if not e.imag]))
    lines = factor_degenerate(B - e * A, zero)
    points = sum([intersections_ellipse_line(A, L, zero) for L in lines], [])
    return points, lines

class NuSol(object):
    def __init__(self, b, mu, ev = None, mW2 = mW**2, mT2 = mT**2, mN2 = mN**2):
        self._b = b
        self._mu = mu
        self.mW2 = mW2
        self.mT2 = mT2
        self.mN2 = mN2
        if ev is None: return
        self.METx = ev.px
        self.METy = ev.py

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
        x = (self.b.M >= 0)*(self.mu.M >= 0)
        return x*np.array([
            [self.Z/math.sqrt(self.Om2)           , 0   , self.x1 - self.mu.mag],
            [self.w * self.Z / math.sqrt(self.Om2), 0   , self.y1              ],
            [0,                                   self.Z, 0                    ]])

    @property
    def R_T(self):
        b_xyz = self.b.x, self.b.y, self.b.z
        R_z = R(2, -self.mu.phi)
        R_y = R(1, 0.5 * math.pi - self.mu.theta)
        R_x = next(R(0, -math.atan2(z, y)) for x, y, z in (R_y.dot(R_z.dot(b_xyz)),))
        return R_z.T.dot(R_y.T.dot(R_x.T))

    @property
    def H(self): return self.R_T.dot(self.BaseMatrix)

    @property
    def X(self):
        S2 = np.vstack([np.vstack([np.linalg.inv([[100, 9], [50, 100]]), [0, 0]]).T, [0, 0, 0]])
        V0 = np.outer([self.METx, self.METy, 0], [0, 0, 1])
        dNu = V0 - self.H
        return np.dot(dNu.T, S2).dot(dNu)

    @property
    def M(self):
        return next(XD + XD.T for XD in (self.X.dot(Derivative()),))

    @property
    def H_perp(self): return np.vstack([self.H[:2], [0, 0, 1]])

    @property
    def N(self):
        hinv = np.linalg.inv(self.H_perp)
        return hinv.T.dot(UnitCircle()).dot(hinv)

def Jacobi(obj, params):
    sml = -1e-9
    for j in range(10000000):
        jbk = []
        res = []
        dres = []
        res_t = None
        v  = obj.residuals(params)
        v_i = sum(v**2)
        for u in range(len(params)):
            dp_ = [x for x in params]
            dp_[u] += sml
            dv = sum(obj.residuals(dp_)**2)
            jbk.append((dv - v_i) / sml)

        jbk = np.array(jbk)
        res = v #np.array(res)
        #if res_t is not None and sum(res) > res_t: delta = -sml * jbk
        delta = -(jbk.T.dot(jbk))**(-1) * jbk.T * v_i
        res_t = sum(res)

        params = params + delta
        print(params, delta, jbk.T, sum(res**2), j)
        if j == 1000: exit()

class SingleNu(NuSol):

    def __init__(self, b, mu, ev):
        t = NuSol(b, mu, ev)
        self._M = t.M
        self._X = t.X
        self._H = t.H

        sols, diag = intersections_ellipses(self._M, UnitCircle())
        self.sols = sorted(sols, key = self.calcX2)

    def calcX2(self, t): return np.dot(t, self._X).dot(t)

    @property
    def chi2(self): return np.array([self.calcX2(self.sols[i]) for i in range(len(self.sols))])

    @property
    def nu(self): return np.array([self._H.dot(self.sols[i]) for i in range(len(self.sols))])

class DoubleNu(NuSol):
    def __init__(self, bs, mus, ev, mT1 = mT, mT2 = mT, mW1 = mW, mW2 = mW):
        b ,  b_ = bs
        mu, mu_ = mus

        sol1 = NuSol(b , mu , None, mW1**2, mT1**2)
        sol2 = NuSol(b_, mu_, None, mW2**2, mT2**2)

        V0 = np.outer([ev.vec.px, ev.vec.py, 0], [0, 0, 1])
        self.S = V0 - UnitCircle()
        N, N_ = sol1.N, sol2.N
        n_ = self.S.T.dot(N_).dot(self.S)
        n  = self.S.T.dot(N ).dot(self.S)

        v, _ = intersections_ellipses(N, n_)
        v_ = [self.S.dot(sol) for sol in v]
        self.solutionSets = [sol1, sol2]
        self.es = [ss.H_perp for ss in self.solutionSets]
        self.met = np.array([ev.vec.px, ev.vec.py, 1])
    
        self.lsq = False
        if not v and leastsq:
            #es = [ss.H_perp for ss in self.solutionSets]
            #met = np.array([ev.vec.px, ev.vec.py, 1])
            #def nus(ts): return tuple(e.dot([math.cos(t), math.sin(t), 1]) for e, t in zip(es, ts))
            #def residuals(params): return sum(nus(params), -met)[:2]
            #ts, _ = leastsq(residuals, [0, 0], ftol=5e-5, epsfcn=0.01)
            #v, v_ = [[i] for i in nus(ts)]
            self.lsq = True
        for k, v in {"perp": v, "perp_": v_, "n_": n_, "n" : n}.items(): setattr(self, k, v)

    def nus(self, ts): return tuple(e.dot([math.cos(t), math.sin(t), 1]) for e, t in zip(self.es, ts))
    def residuals(self, params): return sum(self.nus(params), -self.met)[:2]

    @property
    def nunu_s(self):
        """Solution pairs for neutrino momenta"""

        pairs = {}
        for s, s_ in zip(self.perp, self.perp_):
            pairs[(np.dot(s.T , self.n_).dot(s) - np.dot(s_.T, self.n).dot(s_))**2] = [s, s_]
        pr = sorted(pairs)
        K, K_ = [ss.H.dot(np.linalg.inv(ss.H_perp)) for ss in self.solutionSets]
        nu1 = np.array([K.dot(pairs[s][0])  for s in pr])
        nu2 = np.array([K_.dot(pairs[s][1]) for s in pr])
        self.pairs = pairs
        return nu1, nu2
