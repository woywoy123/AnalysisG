from scipy.spatial.distance import cdist
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import numpy as np
import vector
import math

def dot(v1, v2): return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
def mag(v): return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
def UnitCircle(): return np.diag([1, 1, -1])
def multisqrt(y): return [] if y < 0 else [0] if y == 0 else (lambda r: [-r, r])(math.sqrt(y))

def costheta(v1, v2):
    v1_sq = v1.x**2 + v1.y**2 + v1.z**2
    if v1_sq == 0: return 0
    v2_sq = v2.x**2 + v2.y**2 + v2.z**2
    if v2_sq == 0: return 0
    v1v2 = v1.x*v2.x + v1.y*v2.y + v1.z*v2.z
    return v1v2/math.sqrt(v1_sq * v2_sq)


def R(axis, angle):
    """Rotation matrix about x(0),y(1), or z(2) axis"""
    c, s = math.cos(angle), math.sin(angle)
    R = c * np.eye(3)
    for i in [-1, 0, 1]: R[(axis - i) % 3, (axis + i) % 3] = i * s + (1 - i * i)
    return R

def Derivative():
    """Matrix to differentiate [cos(t),sin(t),1]"""
    cx = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    return cx.dot(np.diag([1, 1, 0]))

def intersections_ellipse_line(ellipse, line, zero=1e-10):
    """Points of intersection between ellipse and line"""
    _, V = np.linalg.eig(np.cross(line, ellipse).T)
    sols = sorted([(
        v.real / v[2].real, np.dot(line, v.real)**2 + np.dot(v.real, ellipse).dot(v.real)**2
        ) for v in V.T if v[2].real != 0 and not sum(v.imag)], key = lambda k : k[1])[:2]
    return [s for s, k in sols if k < zero]

def cofactor(A, i, j):
    """Cofactor[i,j] of 3x3 matrix A"""
    a = A[
        not i : 2 if i == 2 else None : 2 if i == 1 else 1,
        not j : 2 if j == 2 else None : 2 if j == 1 else 1,
    ]
    return (-1) ** (i + j) * (a[0, 0] * a[1, 1] - a[1, 0] * a[0, 1])

def factor_degenerate(G, zero=0):
    """Linear factors of degenerate quadratic polynomial"""
    if G[0, 0] == 0 == G[1, 1]: return [[G[0, 1], 0, G[1, 2]], [0, G[0, 1], G[0, 2] - G[1, 2]]], 0
    swapXY = abs(G[0, 0]) > abs(G[1, 1])
    Q = G[(1, 0, 2),][:, (1, 0, 2)] if swapXY else G
    Q /= Q[1, 1]
    q22 = cofactor(Q, 2, 2)
    if -q22 <= zero: 
        q0 = -cofactor(Q, 0, 0)
        lines = [[Q[0, 1], Q[1, 1], (Q[1, 2] + s)] for s in multisqrt(q0)]
    else:
        q0 = -q22
        x0, y0 = [cofactor(Q, i, 2) / q22 for i in [0, 1]]
        lines = [ [m, Q[1, 1], (-Q[1, 1] * y0 - m * x0)] for m in [Q[0, 1] + s for s in multisqrt(q0)] ]
    return [[L[swapXY], L[not swapXY], L[2]] for L in lines], q0

def intersections_ellipses(A, B, zero = 1e-10):
    """Points of intersection between two ellipses"""
    LA = np.linalg
    sw = abs(LA.det(B)) > abs(LA.det(A))
    if sw: A, B = B, A
    t = LA.inv(A).dot(B)
    e = next(iter([e.real for e in LA.eigvals(t) if not e.imag]))
    lines, q22 = factor_degenerate(B - e * A, zero)
    points = sum([intersections_ellipse_line(A, L, zero) for L in lines], [])
    return points, lines, q22



def get_mw(sl):
    eb, em, bb, bm = sl.b.e, sl.mu.e, sl.b.beta, sl.mu.beta
    mb, mm, mt     = sl.b.tau, sl.mu.tau, sl.mT2
    sin_theta = sl.s
    cos_theta = sl.c

    w = (bm / bb - cos_theta) / sin_theta
    om2 = w**2 + 1 - bm**2
    
    e0 = mm**2 / (2 * em)
    e1 = -1 / (2 * em)
    
    p1 = 1 / (2 * eb)
    p0 = (mb**2 - mt) / (2 * eb)
    sx = e0 - mm**2 / em 
    
    sy0 = (p0 / bb - cos_theta * sx) / sin_theta
    sy1 = (p1 / bb - cos_theta * e1) / sin_theta
    
    x0 = sx * (1 - 1/om2) - (w * sy0) / om2
    x1 = e1 * (1 - 1/om2) - (w * sy1) / om2
    
    d0 = sy0 - w * sx
    d1 = sy1 - w * e1
    
    # quadratic coefficients for z2 = a*v² + b*v + c
    a_val = om2 * x1**2 - d1**2 + e1**2
    b_val = 2 * (om2 * x0 * x1 - d0 * d1) + 2 * e0 * e1 - bm**2
   
    v_crit, v_infl = 0, 0
    v_crit = -b_val / (2 * (a_val if a_val != 0 else 1))
    v_infl = -b_val / (6 * (a_val if a_val != 0 else 1))
    if v_crit >= 0: v_crit = math.sqrt(v_crit)
    else: v_crit = math.sqrt(sl.mW2)
    if v_infl >= 0: v_infl = math.sqrt(v_infl)
    else: v_infl = math.sqrt(sl.mW2)
    return v_crit, v_infl


def get_mt2(sl, timeout = None):
    eb, em, bb, bm   = sl.b.e, sl.mu.e, sl.b.beta, sl.mu.beta
    mb, mm, mt1, mw1 = sl.b.tau, sl.mu.tau, math.sqrt(sl.mT2), math.sqrt(sl.mW2)
    prm = get_mw(sl)
    mw2 = max(prm)
    #if mw2 < 0: return sl
    sin_th = sl.s
    cos_th = sl.c

    w = (bm / bb - cos_th) / sin_th
    omega = w**2 + 1 - bm**2

    x0   = -(mw1**2 - mm) / (2 * em)
    sx   = x0 - em * (1 - bm**2) 
    x0p  = -(mt1**2 - mw1**2 - mb) / (2 * eb)
    sy   = (x0p / bb - cos_th * sx) / sin_th
    x1   = sx - (sx + w * sy) / omega
    z2y  = (x1**2 * omega) - (sy - w * sx)**2 - (mw1**2 - x0**2 - mw1**2 * (1 - bm**2))

    x0_   = -(mw2**2 - mm) / (2 * em)
    sx_   = x0_ - em * (1 - bm**2)
    eps2_ = mw2**2 * (1 - bm**2)
    cons  = mw2**2 - x0_**2 - eps2_

    a_sy = -1 / (2 * eb * bb * sin_th)
    b_sy = ((mw2**2 + mb) / (2 * eb * bb) - cos_th * sx_) / sin_th

    a_x1 = - (w * a_sy) / omega
    b_x1 = ((omega - 1) * sx_ - w * b_sy) / omega
    a    = omega * a_x1**2 - a_sy**2
    b    = 2 * omega * a_x1 * b_x1 - 2 * a_sy * (b_sy - w * sx_)
    c    = omega * b_x1**2 - (b_sy - w * sx_)**2 - cons- z2y
    discriminant = b**2 - 4 * a * c

    root1 = (-b + math.sqrt(discriminant)) / (2 * a)
    root2 = (-b - math.sqrt(discriminant)) / (2 * a)
   
    root1 = math.sqrt(root1) if root1 >= 0 else mt1
    root2 = math.sqrt(root2) if root2 >= 0 else mt1
    mot = min(root1, root2)
    out = NuSol(sl.b, sl.mu, mw2**2, mot**2)
    if timeout is None: timeout = 0
    if abs(mt1 - mot) > 0.1 and timeout < 0: return get_mt2(out, timeout+1)
    return out

def solve_quartic(a, b, c, d, e):
    a = 1.0 / a
    a, b, c, d, e = 1.0, b*a, c*a, d*a, e*a

    b2 = b * b
    b3 = b2 * b
    p = c - (3 * b2) / 8.0
    q = d - (b * c) / 2.0 + b3 / 8.0
    r = e - (3 * b3 * b) / 256.0 + (b2 * c) / 16.0 - (b * d) / 4.0
    s = b / 4.0

    tol = 1e-12
    if abs(q) < tol:
        disc_inner = p * p - 4 * r
        sqrt_disc_inner = disc_inner**0.5
        z1 = (-p + sqrt_disc_inner) / 2.0
        z2 = (-p - sqrt_disc_inner) / 2.0
        return [z1**0.5 - s, -z1**0.5 - s, z2**0.5 - s, -z2**0.5 - s]

    z0 = _solve_cubic(2.0 * p, p * p - 4.0 * r, -q * q, tol)[0]
    m = z0**0.5
    disc1 = m * m - 4.0 * ((p + z0 - q / m) * 0.5)
    disc2 = m * m - 4.0 * ((p + z0 + q / m) * 0.5)
    sqrt_disc1 = disc1**0.5
    sqrt_disc2 = disc2**0.5
    return [i.real for i in [
            (-m + sqrt_disc1) / 2.0 - s, (-m - sqrt_disc1) / 2.0 - s, 
            ( m + sqrt_disc2) / 2.0 - s, ( m - sqrt_disc2) / 2.0 - s
    ] if not i.imag]


def _solve_cubic(a0, a1, a2, tol=1e-12):
    s = a0 / 3.0
    p_cubic = a1 - a0 * a0 / 3.0
    q_cubic = a2 - (a0 * a1) / 3.0 + (2 * a0 ** 3) / 27.0
    if abs(p_cubic) < tol and abs(q_cubic) < tol: return [-s, -s, -s]
    if abs(p_cubic) < tol:
        w0 = (-q_cubic) ** (1/3)
        omega = complex(-0.5, 0.5 * (3**0.5))
        omega2 = complex(-0.5, -0.5 * (3**0.5))
        return [w0 - s, w0 * omega - s, w0 * omega2 - s]

    D = (q_cubic / 2.0) ** 2 + (p_cubic / 3.0) ** 3
    u = -q_cubic / 2.0 + D**0.5
    S  = 0.0 if abs(u) < tol else u ** (1/3)
    w0 = 0.0 if abs(S) < tol else S - p_cubic / (3.0 * S)
    disc_quad = w0 * w0 - 4 * (w0 * w0 + p_cubic)
    sqrt_disc_quad = disc_quad**0.5
    w1 = (-w0 + sqrt_disc_quad) * 0.5
    w2 = (-w0 - sqrt_disc_quad) * 0.5
    return [w0 - s, w1 - s, w2 - s]






class NuSol:
    def __init__(self, b, mu, mW2 = 0, mT2 = 0, mN2 = 0):
        try: self.b  = vector.obj(px =  b[0], py =  b[1], pz =  b[2], E =  b[3]) 
        except: self.b = b
        try: self.mu = vector.obj(px = mu[0], py = mu[1], pz = mu[2], E = mu[3]) 
        except: self.mu = mu

        self.mW2 = mW2
        self.mN2 = mN2
        self.mT2 = mT2
        self.c   = costheta(self.b, self.mu)
        self.s   = math.sqrt(1 - self.c**2)
        self.x0p = -(self.mT2 - self.mW2 - self.b.tau2)/(2*self.b.e)
        self.x0  = -(self.mW2 - self.mu.tau2 - self.mN2)/(2*self.mu.e)
        self.Sx  = (self.x0 * self.mu.beta - self.mu.mag * (1 - self.mu.beta**2))/self.mu.beta**2
        self.Sy  = ((self.x0p / self.b.beta) - self.c * self.Sx) / self.s
        self.w    = (self.mu.beta/self.b.beta - self.c)/self.s
        self.Om2  = self.w**2 + 1 - self.mu.beta**2
        self.eps2 = (self.mW2 - self.mN2) * (1 - self.mu.beta**2)
        self.x1   = self.Sx - ( self.Sx + self.w * self.Sy)/self.Om2
        self.y1   = self.Sy - ( self.Sx + self.w * self.Sy)*self.w/self.Om2
        self.Z2   = self._Z2
        self.R_T  = self._R_T
        self.Z    = math.sqrt(max(1e-31, self.Z2)) if self.Z2 >= 0 else -math.sqrt(abs(self.Z2))
        self.H    = self.R_T.dot(self._BaseMatrix)
        self.H_perp = np.vstack([self.H[:2], [0, 0, 1]])
        self.N    = self._N
    
    @property
    def _Z2(self):
        p1 = (self.x1**2)* self.Om2
        p2 = - (self.Sy - self.w * self.Sx)**2
        p3 = - (self.mW2 - self.x0**2 - self.eps2)
        return  p1 + p2 + p3

    @property
    def _BaseMatrix(self):
        return np.array([
            [math.sqrt(abs(self.Z2)/self.Om2)           , 0     , self.x1 - self.mu.mag],
            [self.w * math.sqrt(abs(self.Z2)/self.Om2)  , 0     , self.y1              ],
            [                                         0 , self.Z, 0                    ]])

    @property
    def _R_T(self):
        self._phi = -self.mu.phi
        self._theta = 0.5*math.pi - self.mu.theta
        b_xyz = self.b.x, self.b.y, self.b.z
        R_z = R(2, -self.mu.phi)
        R_y = R(1, 0.5 * math.pi - self.mu.theta)
        R_x = next(R(0, -math.atan2(z, y)) for x, y, z in (R_y.dot(R_z.dot(b_xyz)),))
        self._alpha = next(-math.atan2(z, y) for x, y, z in (R_y.dot(R_z.dot(b_xyz)),))
        return R_z.T.dot(R_y.T.dot(R_x.T))

    @property
    def _N(self):
        try: hinv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError: return UnitCircle()
        return hinv.T.dot(UnitCircle()).dot(hinv)


    def x_position(self, theta):
        r = self.R_T
        z = self.Z
        o = math.sqrt(self.Om2)
        d1 = (r[0][0] + r[0][1]*self.w) * (z/o)*math.cos(theta)
        d2 = r[0][2]*z*math.sin(theta)
        d3 = (self.x1 - self.mu.mag)*r[0][0] + r[0][1]*self.y1
        return d1 + d2 + d3

    def y_position(self, theta):
        r = self.R_T
        z = self.Z
        o = math.sqrt(self.Om2)
        d1 = (r[1][0] + r[1][1]*self.w) * (z/o)*math.cos(theta)
        d2 = r[1][2]*z*math.sin(theta)
        d3 = (self.x1 - self.mu.mag)*r[1][0] + r[1][1]*self.y1
        return d1 + d2 + d3

    def z_position(self, theta):
        r = self.R_T
        z = self.Z
        o = math.sqrt(self.Om2)
        d1 = (r[2][0] + r[2][1]*self.w) * (z/o)*math.cos(theta)
        d2 = r[2][2]*z*math.sin(theta)
        d3 = (self.x1 - self.mu.mag)*r[2][0] + r[2][1]*self.y1
        return d1 + d2 + d3


    def dist2(self, sl, theta1, theta2):
        x1, y1, z1 = self.x_position(theta1), self.y_position(theta1), self.z_position(theta1)
        x2, y2, z2 =   sl.x_position(theta2),   sl.y_position(theta2),   sl.z_position(theta2)
        return (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2

    def dD_dM(self, sl, theta1, theta2):
        dxt1, dyt1, dzt1 = self.dx_mt(theta1), self.dy_mt(theta1), self.dz_mt(theta1)
        dxw1, dyw1, dzw1 = self.dx_mw(theta1), self.dy_mw(theta1), self.dz_mw(theta1)
        x1  ,   y1,   z1 = self.x_position(theta1), self.y_position(theta1), self.z_position(theta1)

        dxt2, dyt2, dzt2 = sl.dx_mt(theta2),   sl.dy_mt(theta2),   sl.dz_mt(theta2)
        dxw2, dyw2, dzw2 = sl.dx_mw(theta2),   sl.dy_mw(theta2),   sl.dz_mw(theta2)
        x2  ,   y2,   z2 = sl.x_position(theta2), sl.y_position(theta2), sl.z_position(theta2)

        dt_1 = 2 * ((x1-x2) * dxt1 + (y1 - y2) * dyt1 + (z1 - z2)) * dzt1
        dt_2 = 2 * ((x2-x1) * dxt2 + (y2 - y1) * dyt2 + (z2 - z1)) * dzt2

        dw_1 = 2 * ((x1-x2) * dxw1 + (y1 - y2) * dyw1 + (z1 - z2)) * dzw1
        dw_2 = 2 * ((x2-x1) * dxw2 + (y2 - y1) * dyw2 + (z2 - z1)) * dzw2


        return (dt_1, dw_1, dt_2, dw_2)


    def dx_mt(self, theta): return self.Dim_dt(0, theta)
    def dy_mt(self, theta): return self.Dim_dt(1, theta)
    def dz_mt(self, theta): return self.Dim_dt(2, theta)

    def dx_mw(self, theta): return self.Dim_dw(0, theta)
    def dy_mw(self, theta): return self.Dim_dw(1, theta)
    def dz_mw(self, theta): return self.Dim_dw(2, theta)


    def Dim_dt(self, dim, theta):
        r, o = self.R_T, math.sqrt(self.Om2)
        dz_dt, dz_dw, dsx_dw, dsx_dt, dsy_dw, dsy_dt = self.DzDm
        v1 = (r[dim][0] + r[dim][1]*self.w)*(math.cos(theta)/o)*dz_dt 
        v2 = r[dim][2]*math.sin(theta)*dz_dt
        v3 = r[dim][0]*self.dx1_dt + r[dim][1]*self.dy1_dt
        return v1 + v2 + v3

    def Dim_dw(self, dim, theta):
        r, o = self.R_T, math.sqrt(self.Om2)
        dz_dt, dz_dw, dsx_dw, dsx_dt, dsy_dw, dsy_dt = self.DzDm
        v1 = (r[dim][0] + r[dim][1]*self.w)*(math.cos(theta)/o)*dz_dw
        v2 = r[dim][2]*math.sin(theta)*dz_dw
        v3 = r[dim][0]*self.dx1_dw + r[dim][1]*self.dy1_dw
        return v1 + v2 + v3

    @property
    def dx1_dt(self):
        dz_dt, dz_dw, dsx_dw, dsx_dt, dsy_dw, dsy_dt = self.DzDm
        return dsx_dt - (dsx_dt + self.w*dsy_dt)*self.Om2**-1

    @property
    def dy1_dt(self):
        dz_dt, dz_dw, dsx_dw, dsx_dt, dsy_dw, dsy_dt = self.DzDm
        return dsy_dt - (dsx_dt + self.w*dsy_dt)*self.w*self.Om2**-1

    @property
    def dx1_dw(self):
        dz_dt, dz_dw, dsx_dw, dsx_dt, dsy_dw, dsy_dt = self.DzDm
        return dsx_dw - (dsx_dw + self.w*dsy_dw)*self.Om2**-1

    @property
    def dy1_dw(self):
        dz_dt, dz_dw, dsx_dw, dsx_dt, dsy_dw, dsy_dt = self.DzDm
        return dsy_dw - (dsx_dw + self.w*dsy_dw)*self.w*self.Om2**-1

    @property
    def DzDm(self):
        try: return self.deriv
        except: pass

        Z = self.Z2
        Z = -math.sqrt(abs(Z)) if Z < 0 else math.sqrt(Z)
        if Z == 0: Z = 1

        pb  = np.array([self.b.px , self.b.py , self.b.pz])
        pmu = np.array([self.mu.px, self.mu.py, self.mu.pz])
        Eb, Emu = self.b.e  , self.mu.e
        mb, mmu = self.b.tau, self.mu.tau
        beta_b, beta_mu = self.b.beta, self.mu.beta
        mT, mW = math.sqrt(self.mT2), math.sqrt(self.mW2)

        # First derivatives
        dx0p_dt = -mT / Eb
        dx0_dw  = -mW / Emu
      
        dSx_dw = (dx0_dw * beta_mu) / beta_mu**2
        dSy_dt = -mT / (Eb*beta_b*self.s)
        dSy_dw = (mW / (Eb*beta_b) - self.c * dSx_dw) / self.s
        
        dsx = self.Sy - self.w * self.Sx
        dsq = -self.w * dSy_dt/self.Om2
        dsy = dSy_dw - self.w * dSx_dw
        dZ_dmT = 2*(self.Om2 * self.x1 * dsq - dsx * dSy_dt)/Z
        dZ_dmW = 2*(self.Om2 * self.x1 * (dSx_dw * (self.Om2 - 1) - self.w * dSy_dw) * self.Om2**-1 - dsx * dsy - mW + self.x0 * dx0_dw + (1 - beta_mu**2) * mW)/Z
        self.deriv = (dZ_dmT, dZ_dmW, dSx_dw, 0, dSy_dw, dSy_dt)
        return self.deriv

    @property
    def ellipse_property(self):
        A = self.H[:, 0]
        B = self.H[:, 1]
        
        N = np.cross(A, B)
        norm_N = np.linalg.norm(N)
        N_normalized = N / norm_N if norm_N > 1e-10 else N
        
        M = np.array([[np.dot(A, A), np.dot(A, B)], 
                      [np.dot(A, B), np.dot(B, B)]])
        
        trace = M[0,0] + M[1,1]
        det   = M[0,0] * M[1,1] - M[0,1]**2
        dsc = np.sqrt(trace**2 - 4*det)
        l1, l2 = (trace + dsc)/2, (trace - dsc)/2
        
        major, minor = np.sqrt(max(l1, l2)), np.sqrt(min(l1, l2))
        area = np.pi * major * minor
        
        return {
            'centroid': self.H[:, 2],
            'normal': N_normalized,
            'semi_major': major,
            'semi_minor': minor,
            'area': area
        }

    @property
    def angle_z(self): return np.acos(np.dot(self.ellipse_property["normal"], [0, 0, 1]))
    @property
    def angle_y(self): return np.acos(np.dot(self.ellipse_property["normal"], [0, 1, 0]))
    @property
    def angle_x(self): return np.acos(np.dot(self.ellipse_property["normal"], [1, 0, 0]))
    @property
    def OptimizeMW(self): 
        mw2 = max(get_mw(self))
        return NuSol(self.b, self.mu, mw2**2, self.mT2)
    @property
    def OptimizeMT(self): return get_mt2(self)

    @staticmethod
    def get_intersection_angle(H1, H2, lim = 1e-6):
        H1, H2 = H1.tolist(), H2.tolist()
        d1 = H2[0][2] - H1[0][2]
        d2 = H2[1][2] - H1[1][2]
        x1 = H1[0][0]*H1[1][1] - H1[0][1]*H1[1][0]
        x2 = H2[0][0]*H2[1][1] - H2[0][1]*H2[1][0]
        if abs(x1) < lim or abs(x2) < lim: return []

        x1, x2 = x1**-1, x2**-1
        h1_00, h1_01 =  H1[1][1]*x1, -H1[0][1]*x1
        h1_10, h1_11 = -H1[1][0]*x1,  H1[0][0]*x1
        w00, w01 = h1_00*H2[0][0] + h1_01 * H2[1][0], h1_00*H2[0][1] + h1_01 * H2[1][1]
        w10, w11 = h1_10*H2[0][0] + h1_11 * H2[1][0], h1_10*H2[0][1] + h1_11 * H2[1][1]
        r00, r11 = h1_00*d1 + h1_01*d2, h1_10*d1 + h1_11*d2

        p  = w01**2 + w11**2 - w00**2 - w10**2
        q  = 2*(w00*w01 + w10*w11)
        rv = 2*(w00*r00 + w10*r11)
        sv = 2*(w01*r00 + w11*r11)
        tv = r00**2 + r11**2 - 1 + w00**2 + w10**2

        a = p**2 + q**2
        b = 2*(q*sv - p*rv)
        c = sv**2 - 2*p*(p+tv) + rv**2 - q**2
        d = 2*(rv * (p+tv)-q*sv)
        e = (p+tv)**2 - sv**2
        roots = solve_quartic(a, b, c, d, e)
        solutions = []
        for u in roots:
            if -1 > u or u > 1: continue
            x = 1 - u**2
            v_pos =  x**0.5 if x >= 0 else None
            v_neg = -x**0.5 if x >= 0 else None
            for v in [v_neg, v_pos]:
                if v is None: continue
                x1 = w00 * u + w01 * v + r00
                y1 = w10 * u + w11 * v + r11
                if abs(x1**2 + y1**2 - 1) > lim: continue
                lhs = (H1[2][0] * x1 + H1[2][1] * y1 + H1[2][2])
                rhs = (H2[2][0] * u  + H2[2][1] * v  + H2[2][2])
                if abs(lhs - rhs) > lim: continue
                solutions += [(np.arctan2(y1, x1).item(), np.arctan2(v, u).item())]
        return solutions 

        

def figs():
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('3D Ellipses with Circle and Intersection Plane', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.view_init(elev=90, azim=-90)
    return ax

def plot_ellipse(ellipse, ax, label, color, lin = None):
    t = np.linspace(0, 2*np.pi, 1000)
    ar = _make_ellipse(ellipse, t)
    ax.plot(ar[:,0], ar[:,1], ar[:,2], label = label, color = color, linestyle = lin, linewidth=2, alpha=0.9)

def plane_equation(params, points):
    a, b, c, d = params
    x, y, z = points.T
    return np.abs(a*x + b*y + c*z + d) / np.sqrt(a**2 + b**2 + c**2)

def fit_plane(points):
    centroid = np.mean(points, axis=0)
    def residuals(params): return plane_equation(params, points - centroid)
    result = least_squares(residuals, [1, 1, 1, 0])
    a, b, c, d = result.x
    normal = np.array([a, b, c])
    norm = np.linalg.norm(normal)
    if norm <= 1e-10: return normal, d
    normal /= norm
    d = d / norm + np.dot(normal, centroid)
    return normal, d


def _make_ellipse(ellipse, t):
    return np.array([
        ellipse[0,0]*np.cos(t) + ellipse[0,1]*np.sin(t) + ellipse[0,2],
        ellipse[1,0]*np.cos(t) + ellipse[1,1]*np.sin(t) + ellipse[1,2],
        ellipse[2,0]*np.cos(t) + ellipse[2,1]*np.sin(t) + ellipse[2,2]
    ])
 
def find_intersection_points(ellipse1, ellipse2, num_points=1000):
    t = np.linspace(0, 2*np.pi, 10)
    pts1, pts2  = _make_ellipse(ellipse1, t), _make_ellipse(ellipse2, t)
    dist_matrix = cdist(_make_ellipse(ellipse1, t), _make_ellipse(ellipse2, t))
    intersection_points = []

    pt1, pt2 = _make_ellipse(ellipse1, t), _make_ellipse(ellipse2, t)
    print(pt1 - pt2)

    for i in range(10):
        for j in range(10):
            print(sum((pt1[:,i] - pt2[:,j])**2)**0.5)
    print(cdist(_make_ellipse(ellipse1, t), _make_ellipse(ellipse2, t)))

    exit()

    tol = sorted([np.min(dist_matrix[i]) for i in range(num_points)])
    tol = np.array(tol[:10]).mean()
    for i in range(num_points):
        min_dist = np.min(dist_matrix[i])
        if min_dist > tol: continue
        idy = np.argmin(dist_matrix[i])
        intersection_points.append((pts1[i] + pts2[idy])/2)
    return np.array(intersection_points)

def generate_plane(ellipse1, ellipse2):
    intersection = find_intersection_points(ellipse1, ellipse2, num_points=1000)
    normal, plane_d = fit_plane(intersection)
    print(f"Plane equation: {normal[0]:.6f}*x + {normal[1]:.6f}*y + {normal[2]:.6f}*z + {plane_d:.6f} = 0")
   
    centroid = np.mean(intersection, axis=0)
    basis1 = np.cross(normal, [1, 0, 0])
    if np.linalg.norm(basis1) < 1e-5: basis1 = np.cross(normal, [0, 1, 0])
    basis1 /= np.linalg.norm(basis1)
    basis2 = np.cross(normal, basis1)
    basis2 /= np.linalg.norm(basis2)
   
    print(f"centroid: x = {centroid[0]:.6f}; y = {centroid[1]:.6f}; z = {centroid[2]:.6f}")
    size = 200
    x_grid, y_grid = np.meshgrid(np.linspace(-size, size, 10), np.linspace(-size, size, 10))
    plane_points = [[centroid + x_grid[i,j]*basis1 + y_grid[i,j]*basis2 for j in range(10)] for i in range(10)]
    return np.array(plane_points)

def plot_plane(A1 = None, A2 = None, plane_normal = None, points_ = None, ax = None, label = "plane", size = 2000):
    if A1 is not None: points = generate_plane(A1, A2)
    else: points = points_

    #x, y = np.meshgrid(np.linspace(-size, size, 10), np.linspace(-size, size, 10))
    #points = np.zeros((10, 10, 3))

    #basis1 = np.cross(plane_normal, [1, 0, 0])
#    if np.linalg.norm(basis1) < 1e-5: basis1 = np.cross(plane_normal, [0, 1, 0])
    #basis1 /= np.linalg.norm(basis1)
    #basis2 = np.cross(plane_normal, basis1)
    #basis2 /= np.linalg.norm(basis2)

    #for i in range(10):
    #    for j in range(10):
    #        points[i, j] = points_ + x[i,j]*basis1 + y[i,j]*basis2
    ax.plot_surface(points[:, :, 0], points[:, :, 1], points[:, :, 2], alpha=0.3, color='green', label = label)

def plot():
    plt.tight_layout()
    plt.show() 

