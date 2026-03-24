import numpy as np

def omega(data, sign): return (1.0 / data.theta.sin) * (sign * (data.b_mu / data.b_b) - data.theta.cos)
def Omega(data, sign): return (omega(data, sign) ** 2 + 1 - data.b_mu ** 2)**0.5
def Gamma(data, sign): return (omega(data, +1) + sign * omega(data, -1)) / Omega(data, sign) ** 2 

def delta(data, sign):
    wp = omega(data, +1); wm = omega(data, -1)
    op = Omega(data, +1); om = Omega(data, -1)
    a = 1 - data.b_mu ** 2 - wp * wm + sign*(op * om)
    return a / (wp + wm)

def Z2_coeffs(data, sign):
    w, o = omega(data, sign), Omega(data, sign)
    b_mu, m_mu  = data.b_mu, data.m_mu
    a = (b_mu ** 2 - w ** 2) / o**2
    b = 2 * w  / o**2
    c = - (1 - b_mu**2) / o**2
    d = 2 * data.p_mu
    e = m_mu ** 2 
    return [a, b, c, d, e]

def Sx0(data): return - (data.m_mu ** 2) / (data.p_mu)
def Sy0(data, sign): return - omega(data, sign) * data.e_mu / data.b_mu

def costheta(v1, v2):
    v1_sq = v1.px**2 + v1.py**2 + v1.pz**2
    if v1_sq == 0: return 0
    v2_sq = v2.px**2 + v2.py**2 + v2.pz**2
    if v2_sq == 0: return 0
    v1v2 = v1.px*v2.px + v1.py*v2.py + v1.pz*v2.pz
    return v1v2/np.sqrt(v1_sq * v2_sq)

def find_intersection(L1, L2):
    mag1 = np.linalg.norm(L1.d)
    mag2 = np.linalg.norm(L2.d)
    if mag1 == 0 or mag2 == 0: return None
        
    d1 = L1.d / mag1
    d2 = L2.d / mag2
    w0 = L1.r0 - L2.r0
    
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d_dp = np.dot(d1, w0)
    e = np.dot(d2, w0)
    
    D = a * c - b * b 
    if D < 1e-8: return None
        
    t = (b * e - c * d_dp) / D
    u = (a * e - b * d_dp) / D
    
    p1_intersect = L1.r0 + t * d1
    p2_intersect = L2.r0 + u * d2
    return (p1_intersect + p2_intersect) / 2.0
