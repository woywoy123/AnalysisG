import numpy as np
import math

def string(obj, tl, mrg = 8): 
    d = str(getattr(obj, tl))
    l = "".join([" " for i in range(mrg - len(d))])
    return str(tl) + ": " + l + d

def signs(v1, v2, s): return v1 if s > 0 else v2

def costheta(v1, v2):
    v1_sq = v1.px**2 + v1.py**2 + v1.pz**2
    if v1_sq == 0: return 0
    v2_sq = v2.px**2 + v2.py**2 + v2.pz**2
    if v2_sq == 0: return 0
    v1v2 = v1.px*v2.px + v1.py*v2.py + v1.pz*v2.pz
    return v1v2/math.sqrt(v1_sq * v2_sq)

def rotation_z(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
def rotation_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
def rotation_x(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def signs(s, v1, v2): return v1 if s > 0 else v2

def omega(data, sign): return (1 /data.theta.sin) * (sign * (data.b_mu / data.b_b) - data.theta.cos)
def Omega(data, sign): return (omega(data, sign) ** 2 + 1 - data.b_mu ** 2)**0.5

def x1(sx, sy, data, sign): return sx - (sx + omega(data, sign) * sy) / Omega(data, sign)**2 
def y1(sx, sy, data, sign): return sy - (sx + omega(data, sign) * sy) * omega(data, sign) / Omega(data, sign)**2 

def H_tilde(sx, sy, sz, data, sign):
    x = np.array([
        [sz / Omega(data, sign),                     0, x1(sx, sy, data, sign) - data.p_mu], 
        [omega(data, sign) * sz / Omega(data, sign), 0, y1(sx, sy, data, sign)            ],
        [0                                         , sz, 0                                ]
    ]).real
    return x

def H2_tilde(sx, sy, sz2, data, sign):
    o2, w = Omega(data, sign)**2, omega(data, sign)
    _x1, _y1 = x1(sx, sy, data, sign), y1(sx, sy, data, sign)
    return np.array([
        [sz2 / o2 + (_x1 - data.p_mu)**2       , sz2 * w / o2 + _y1 * (_x1 - data.p_mu),   0],
        [sz2 * w / o2 + _y1 * (_x1 - data.p_mu), sz2 * w ** 2 /o2 + _y1 ** 2           ,   0],
        [0                                     , 0                                     , sz2]
    ])


def nu_mass(data, s1, s2):
    d = delta(data, s2)
    w = omega(data, s1)
    o = Omega(data, s1)
    b, p, m = data.b_mu, data.p_mu, data.m_mu
    return complex((data.m_mu ** 2 - (p * d * o)**2 / ((b ** 2 - w ** 2)*d**2 + 2 * w * d - (1 - b ** 2 )))*0.5)**0.5

def Gamma(data, sign):
    wp = omega(data, +1)
    wm = omega(data, -1)
    o2 = Omega(data, sign) ** 2 
    return (wp + sign*wm) / o2

def delta(data, sign):
    wp = omega(data, +1); wm = omega(data, -1)
    op = Omega(data, +1); om = Omega(data, -1)
    a = 1 - data.b_mu ** 2 - wp * wm + sign*(op * om)
    return a / (wp + wm)

def mW(data, m_nu, sx):
    return complex(m_nu**2 - data.m_mu**2 - 2 * data.p_mu * sx) ** 0.5

def mT(data, m_nu, sx, sy):
    a = m_nu**2 - data.m_mu**2 + data.m_b**2
    a += -2 * (data.p_mu * sx + data.p_mu * (data.theta.cos * sx + data.theta.sin * sy))
    return complex(a) ** 0.5

class branch_t:
    def __init__(self, v1, v2):
        self.p = v1; self.m = v2

class angular_t:
    def __init__(self, alpha):
        self.cos = math.cos(alpha)
        self.sin = math.sin(alpha)
        self.tan = math.tan(alpha)
        self.alpha = alpha

class hyper_t:
    def __init__(self, tau):
        self.cosh = math.cosh(tau)
        self.sinh = math.sinh(tau)
        self.tanh = math.tanh(tau)

class line_t:

    def __init__(self, S1, S2, signs):
        self.Collinear = True
        self.Intersect = False
        self.signs     = signs

        S1, S2 = list(S1), list(S2)
        self.m_nu1     = S1.pop(0)
        self.m_nu2     = S2.pop(0)
        S1 = [self.to_real(i) if abs(i.imag) > 0 else i for i in S1]
        S2 = [self.to_real(i) if abs(i.imag) > 0 else i for i in S2]
        self.sx1, self.sy1, self.sx2, self.sy2 = [i.real for i in sum([S1, S2], [])]
        self.dy , self.dx = self.sy2 - self.sy1, self.sx2 - self.sx1
        self.b1 = 0; self.b2 = 0
        if abs(self.dx) == 0: return
        self.Collinear = False

        self.m = self.dy / self.dx
        self.b1 = self.sy1 - self.m * self.sx1
        self.b2 = self.sy2 - self.m * self.sx2 
        if abs(self.b1 - self.b2) > 1e-8: return 
        self.Intersect = True
        self.solutions = []

    def __call__(self, t):
        return self.m * t + self.b1 

    def to_real(self, v):
        a, b = v.real, v.imag
        th = math.atan(b / a)
        return (complex(a, b)*math.e**complex(0, -th)).real

def _print(tlt, val = None, obj = None):
    if val is None: return print("====== " + tlt + " ======")
    if isinstance(val, list) and obj is not None: 
        val = sum([string(obj, i) + " \n" for i in val])
    if isinstance(val, dict):
        o = ""
        for i in val: o += string_(i, val[i]) + " \n"
        val = o

    print("-------- " + tlt + " ------")
    print(val)

def assertions(tlt, trgt, val, limit =  0.1, hard_limit = False):
    try: 
        if isinstance(trgt, np.ndarray): 
            assert sum(sum(abs(trgt - val) / abs(trgt + (trgt == 0))*100 > limit)) == 0
            print(tlt + "+>: diff:\n", abs(trgt - val))
            return True
        else: 
            if trgt == 0: assert abs(trgt - val) < limit
            else: assert (abs(trgt - val) / abs(trgt))*100 < limit
        print(tlt + "+>:\n", trgt, val, "diff:", abs(trgt - val))
        return True
    except AssertionError: print(tlt + "!>:\n", trgt, "\n", val, "\ndiff:\n", abs(trgt - val))
    if hard_limit: raise AssertionError

def observation(tlt, val):
    print("NOTE: " + tlt + " +>: ", val)


