from atomics import *

# --- surface parameterization: m_nu = 0.
def p_Sx1(obj, z, tau): return -z * (obj.a_x * cosh(tau) + obj.b_x * sinh(tau)) + obj.c_x
def p_Sy1(obj, z, tau): return -z * (obj.a_y * cosh(tau) + obj.b_y * sinh(tau)) + obj.c_y
def p_Z2(obj, Sx, Sy):  return obj.A * Sx ** 2 + obj.B * Sy ** 2 + obj.C * Sy*Sx  + obj.D * Sx + obj.E

# --- H_tilde
def p_h_tilde(obj, z, tau): return z * (obj.htc + obj.ht1 * cosh(tau) + obj.ht2 * sinh(tau))
def p_hmatrix(obj, z, tau): return z * (obj.hc  +  obj.h1 * cosh(tau) +  obj.h2 * sinh(tau))

def SxSy(obj, z, tau): return obj.cos * obj.Sx(z, tau) + obj.sin * obj.Sy(z, tau)

# --- A_mu:
def p_Amu(obj, z, tau, m_nu): 
    out  = obj.amc
    out += obj.am1 * obj.Sx(z, tau) 
    out += obj.am2 * obj.Sx(z, tau) ** 2 
    out += m_nu
    return out

def p_Ab(obj, z, tau, m_nu):
    out  = obj.bmc
    out += obj.bm1 * SxSy(obj, z, tau)
    out += obj.bm2 * obj.Sx(z, tau)
    out += obj.bm3 * SxSy(obj, z, tau) ** 2
    out += m_nu
    return out

# --- geometry
def p_center(mtx): return mtx @ np.array([0, 0, 1])
def p_semi_a(mtx): return mtx @ np.array([1, 0, 0])
def p_semi_b(mtx): return mtx @ np.array([0, 1, 0])
def p_normal(mtx): return np.cross(p_semi_a(mtx), p_semi_b(mtx))


# --- plane equation: n (r - center) = 0
def p_plane(mtx, gen = None):
    n = p_normal(mtx)
    c = p_center(mtx)
    d = np.dot(n, c)
    if gen is None: return n, c, d

    gen.data.normal = n
    gen.data.center = c
    gen.data.r0     = d
    return n, c, d

def p_ellipse(mtx, gen = None): 
    n = p_normal(mtx)
    c = p_center(mtx)
    d = np.dot(n, c)
    if gen is None: return n, c, d
    gen.data.matrix = mtx
    gen.data.center = c
    return n, c, d

