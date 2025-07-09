from AnalysisG.selections.neutrino.validation.validation import Neutrino
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import numpy as np
import math

#mW = 80.385*1000
#mT = 172.62*1000
#mN = 0

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
    for i in [-1, 0, 1]: R[(axis - i) % 3, (axis + i) % 3] = i * s + (1 - i * i)
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
    if G[0, 0] == 0 == G[1, 1]: return [[G[0, 1], 0, G[1, 2]], [0, G[0, 1], G[0, 2] - G[1, 2]]]
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
        lines = [ [m, Q[1, 1], (-Q[1, 1] * y0 - m * x0)] for m in [Q[0, 1] + s for s in multisqrt(-q22)] ]
    return [[L[swapXY], L[not swapXY], L[2]] for L in lines], q0

def intersections_ellipses(A, B, returnLines=False, zero = 10e-10):
    """Points of intersection between two ellipses"""
    LA = np.linalg
    sw = abs(LA.det(B)) > abs(LA.det(A))
    if sw: A, B = B, A
    t = LA.inv(A).dot(B)
    e = next(iter([e.real for e in LA.eigvals(t) if not e.imag]))
    lines, q22 = factor_degenerate(B - e * A, zero)
    points = sum([intersections_ellipse_line(A, L, zero) for L in lines], [])
    return points, lines, q22


def get_mW(sl):
    Eb, Em, Bb, Bm = sl.b.e, sl.mu.e, sl.b.beta, sl.mu.beta
    mb, mm, mT     = sl.b.tau, sl.mu.tau, sl.mT2
    sin_theta = sl.s
    cos_theta = sl.c

    w = (Bm / Bb - cos_theta) / sin_theta
    om2 = w**2 + 1 - Bm**2
    
    E0 = mm**2 / (2 * Em)
    E1 = -1 / (2 * Em)
    
    P1 = 1 / (2 * Eb)
    P0 = (mb**2 - mT) / (2 * Eb)
    Sx = E0 - mm**2 / Em 
    
    Sy0 = (P0 / Bb - cos_theta * Sx) / sin_theta
    Sy1 = (P1 / Bb - cos_theta * E1) / sin_theta
    
    X0 = Sx * (1 - 1/om2) - (w * Sy0) / om2
    X1 = E1 * (1 - 1/om2) - (w * Sy1) / om2
    
    D0 = Sy0 - w * Sx
    D1 = Sy1 - w * E1
    
    # Quadratic coefficients for Z2 = A*v² + B*v + C
    A_val = om2 * X1**2 - D1**2 + E1**2
    B_val = 2 * (om2 * X0 * X1 - D0 * D1) + 2 * E0 * E1 - Bm**2
   
    v_crit, v_infl = 0, 0
    v_crit = -B_val / (2 * (A_val if A_val != 0 else 1))
    v_infl = -B_val / (6 * (A_val if A_val != 0 else 1))
    if v_crit >= 0: v_crit = math.sqrt(v_crit)
    if v_infl >= 0: v_infl = math.sqrt(v_infl)
    return v_crit, v_infl


def get_mT2(sl):
    Eb, Em, Bb, Bm   = sl.b.e, sl.mu.e, sl.b.beta, sl.mu.beta
    mb, mm, mT1, mW1 = sl.b.tau, sl.mu.tau, math.sqrt(sl.mT2), math.sqrt(sl.mW2)
    prm = get_mW(sl)
    mW2 =prm[0]

    sin_th = sl.s
    cos_th = sl.c

    w = (Bm / Bb - cos_th) / sin_th
    Omega = w**2 + 1 - Bm**2

    x0   = -(mW1**2 - mm) / (2 * Em)
    Sx   = x0 - Em * (1 - Bm**2) 
    x0p  = -(mT1**2 - mW1**2 - mb) / (2 * Eb)
    Sy   = (x0p / Bb - cos_th * Sx) / sin_th
    x1   = Sx - (Sx + w * Sy) / Omega
    Z2y  = (x1**2 * Omega) - (Sy - w * Sx)**2 - (mW1**2 - x0**2 - mW1**2 * (1 - Bm**2))

    x0_   = -(mW2**2 - mm) / (2 * Em)
    Sx_   = x0_ - Em * (1 - Bm**2)
    eps2_ = mW2**2 * (1 - Bm**2)
    cons  = mW2**2 - x0_**2 - eps2_

    A_sy = -1 / (2 * Eb * Bb * sin_th)
    B_sy = ((mW2**2 + mb) / (2 * Eb * Bb) - cos_th * Sx_) / sin_th

    A_x1 = - (w * A_sy) / Omega
    B_x1 = ((Omega - 1) * Sx_ - w * B_sy) / Omega
    A    = Omega * A_x1**2 - A_sy**2
    B    = 2 * Omega * A_x1 * B_x1 - 2 * A_sy * (B_sy - w * Sx_)
    C    = Omega * B_x1**2 - (B_sy - w * Sx_)**2 - cons- Z2y
    discriminant = B**2 - 4 * A * C

    root1 = (-B + math.sqrt(discriminant)) / (2 * A)
    root2 = (-B - math.sqrt(discriminant)) / (2 * A)
   
    root1 = math.sqrt(root1) if root1 >= 0 else 0
    root2 = math.sqrt(root2) if root2 >= 0 else 0
    print([root1*0.001, root2*0.001], [i*0.001 for i in prm])
    mot = min(root1, root2)
    out = NuSol(sl.b, sl.mu, None, mW2**2, mot**2)
    if abs(mT1 - mot) > 0.1: return get_mT2(out)
    print("----")
    return out

    #return min(root1, root2)


def compute_center(A, P, n):
    M = A[:2, :2]
    rhs = -A[:2, 2]
    uvc = np.linalg.solve(M, rhs)
    u_vec = np.array([1, 0, 0]) if not np.allclose(n, [1,0,0]) else np.array([0,1,0])
    v_vec = np.cross(n, u_vec)
    u_vec = np.cross(v_vec, n)  # Ensure orthogonality
    u_vec, v_vec = u_vec / np.linalg.norm(u_vec), v_vec / np.linalg.norm(v_vec)
    # Map to 3D
    C = P + uvc[0] * u_vec + uvc[1] * v_vec
    return C

def compute_circle_for_ellipse(points):
    points = np.array(points, dtype=np.float64)
    centroid = np.mean(points, axis=0)
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    distances = np.linalg.norm(points - centroid, axis=1)
    radius = np.mean(distances)
    return centroid, normal, radius

def generate_3d_circle(center, normal, radius, num_points=100):
    n = np.array(normal)
    n_normalized = n / np.linalg.norm(n)

    # Choose an arbitrary vector not parallel to normal
    arbitrary_vec = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(arbitrary_vec, n_normalized)) > 0.999:
        arbitrary_vec = np.array([0.0, 1.0, 0.0])
    if np.abs(np.dot(arbitrary_vec, n_normalized)) > 0.999:
        arbitrary_vec = np.array([0.0, 0.0, 1.0])

    # Project arbitrary vector onto the plane
    proj = arbitrary_vec - np.dot(arbitrary_vec, n_normalized) * n_normalized
    u = proj / np.linalg.norm(proj)
    v = np.cross(n_normalized, u)
    v = v / np.linalg.norm(v)

    # Generate circle points
    theta = np.linspace(0, 2*np.pi, num_points)
    circle_points = center + radius * (np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v))
    return circle_points

class NuSol(object):
    def __init__(self, b, mu, ev = None, mW2 = 0, mT2 = 0, mN2 = 0):
        self._b = b
        self._mu = mu
        self.mW2 = mW2
        self.mN2 = mN2
        self.mT2 = mT2 #self.massT2
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
        self._phi = -self.mu.phi
        self._theta = 0.5*math.pi - self.mu.theta
        b_xyz = self.b.x, self.b.y, self.b.z
        R_z = R(2, -self.mu.phi)
        R_y = R(1, 0.5 * math.pi - self.mu.theta)
        R_x = next(R(0, -math.atan2(z, y)) for x, y, z in (R_y.dot(R_z.dot(b_xyz)),))
        self._alpha = next(-math.atan2(z, y) for x, y, z in (R_y.dot(R_z.dot(b_xyz)),))
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
    def M(self): return next(XD + XD.T for XD in (self.X.dot(Derivative()),))

    @property
    def H_perp(self):  return np.vstack([self.H[:2], [0, 0, 1]])



def plotthis(lf, name, lx = 100):
    u = np.linspace(-2*np.pi, 2 * np.pi, lx)
    c = np.cos(u).reshape((lx, 1))
    s = np.sin(u).reshape((lx, 1))
    o = np.ones((lx, 1))
    xs = np.concatenate((c, s, o), -1)
    xs = xs.reshape((lx, 1, 3))

    col = iter(["b", "y", "r", "grey", "orange"])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
    for i in lf:
        Ap = np.inner(i, xs).reshape(3, lx)
        ax.plot_trisurf(Ap[0], Ap[1], Ap[2], color=next(col), alpha=0.6)
    
    #ax.view_init(elev=90, azim=0) 
 
    # Set labels
#    ax.set_xlim(-2, 2)
#    ax.set_ylim(-2, 2)
#    ax.set_zlim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    plt.savefig(name + ".png", dpi = 301)
    plt.close()


class DoubleNu:

    def __init__(self, bs, mus, ev, mW1, mT1, mW2, mT2, metz, itx = None):
        self.lsq = False
        b ,  b_ = bs
        mu, mu_ = mus 
        self.ev = np.array([ev.px, ev.py, 0])
        self.itx = itx

        print(self.ev)

        self.sol1 = NuSol(b , mu , None, mW1**2, mT1**2, 0)
        self.sol2 = NuSol(b_, mu_, None, mW2**2, mT2**2, 0)

        print("----Truth----")
        self.sol_1 = get_mT2(self.sol1)
        self.sol_2 = get_mT2(self.sol2)

        print("----fixed----")
        mw1 = 82.6*1000
        mw2 = 82.6*1000
        mt1 = 172.16 * 1000
        mt2 = 172.16 * 1000

        print("b1_px = ", b.px  ,"; b1_py = ", b.py  ,"; b1_pz = ", b.pz  ,"; b1_e = ", b.e  , "; b1_m = ", b.tau  )
        print("m1_px = ", mu.px ,"; m1_py = ", mu.py ,"; m1_pz = ", mu.pz ,"; m1_e = ", mu.e , "; m1_m = ", mu.tau  )
        print("b2_px = ", b_.px ,"; b2_py = ", b_.py ,"; b2_pz = ", b_.pz ,"; b2_e = ", b_.e , "; b2_m = ", b_.tau  )
        print("m2_px = ", mu_.px,"; m2_py = ", mu_.py,"; m2_pz = ", mu_.pz,"; m2_e = ", mu_.e, "; m2_m = ", mu_.tau  )

        exit()


        self.dol1 = NuSol(b , mu , None, mw1**2, mt1**2, 0)
        self.dol2 = NuSol(b_, mu_, None, mw2**2, mt2**2, 0)

        self.dol_1 = get_mT2(self.dol1)
        self.dol_2 = get_mT2(self.dol2)

        x0 = ev.px
        y0 = ev.py
        z0 = 0

        theta = np.radians(0)
        phi   = np.radians(0)
        c , s  = np.cos(theta), np.sin(theta)
        c_, s_ = np.cos(phi), np.sin(phi)

        R = np.array([[c_, s_*s, s_*c], [0, c, -s], [-s_, c_*s, c_*c]])
        T = np.array([[R[0][0], R[0][1], 0], [0, R[1][1], 0], [0, 0, 1]])

        self.S = np.outer([ev.px, ev.py, 0], [0, 0, 1]) - UnitCircle()
        self.S = (self.S + self.S.T)/2
        self.S = T.T.dot(self.S).dot(T)

        print(self.S)
        exit()


        lx = 100
        u = np.linspace(0, 2 * np.pi, lx)
        o = np.ones((lx, 1))
        c = np.cos(u).reshape((lx, 1))
        s = np.sin(u).reshape((lx, 1))

        xs = np.concatenate((c, s, o), -1)
        xs = xs.reshape((lx, 1, 3))

        A1 = np.inner(self.sol1.H, xs).reshape(3, lx)
        A2 = np.inner(self.sol2.H, xs).reshape(3, lx)

        B1 = np.inner(self.sol_1.H, xs).reshape(3, lx)
        B2 = np.inner(self.sol_2.H, xs).reshape(3, lx)

        _A1 = np.inner(self.dol1.H, xs).reshape(3, lx)
        _A2 = np.inner(self.dol2.H, xs).reshape(3, lx)

        _B1 = np.inner(self.dol_1.H, xs).reshape(3, lx)
        _B2 = np.inner(self.dol_2.H, xs).reshape(3, lx)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') 

        # Truth
        ax.plot(A1[0], A1[1], A1[2], color="black", alpha=0.6, linewidth = 2.0)
        ax.plot(A2[0], A2[1], A2[2], color="black", alpha=0.6, linewidth = 2.0)

        # Adjusted from Truth
        ax.plot(B1[0], B1[1], B1[2], color="purple", alpha=0.6, linewidth = 2.0)
        ax.plot(B2[0], B2[1], B2[2], color="purple", alpha=0.6, linewidth = 2.0)

        # Fixed point
        ax.plot(_A1[0], _A1[1], _A1[2], color="orange", alpha=0.6, linestyle = ":", linewidth = 2.0)
        ax.plot(_A2[0], _A2[1], _A2[2], color="orange", alpha=0.6, linestyle = "--", linewidth = 2.0)

        # Adjusted fixed point
        ax.plot(_B1[0], _B1[1], _B1[2], color="purple", alpha=0.6, linestyle = ":", linewidth = 2.0)
        ax.plot(_B2[0], _B2[1], _B2[2], color="purple", alpha=0.6, linestyle = "--", linewidth = 2.0)

        #ax.plot(Cx[0], Cx[1], Cx[2], color="red", alpha=0.6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

        exit()

        # Compute circle parameters
        centroid1, normal1, radius1 = compute_circle_for_ellipse(self.sol1.H)
        centroid2, normal2, radius2 = compute_circle_for_ellipse(self.sol2.H)
        
        # Generate 3D circle points
        circle1_points = generate_3d_circle(centroid1, normal1, radius1)
        circle2_points = generate_3d_circle(centroid2, normal2, radius2)


        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot ellipse1 and its circle
        #ax.scatter(*np.array(self.sol1.H).T, c='red', s=50, label='Ellipse 1 Points')
        ax.scatter(*centroid1, c='black', s=100, label='Ellipse 1 Center')
        #ax.plot(*circle1_points.T, 'blue', label='Ellipse 1 Circle')
       
        # Plot ellipse2 and its circle
        #ax.scatter(*np.array(self.sol2.H).T, c='green', s=50, label='Ellipse 2 Points')
#        ax.scatter(*centroid2, c='purple', s=100, label='Ellipse 2 Center')
        #ax.plot(*circle2_points.T, 'orange', label='Ellipse 2 Circle')


        print(A1[0])
        print(A1)
        exit()
        ax.plot(A1[0], A1[1], A1[2], color="black", alpha=0.6, linewidth = 2.0)
#        ax.plot(A2[2], A2[1], A2[0], color="green", alpha=0.6, linewidth = 2.0)

        # Configure plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Visualization of Ellipses and Their Circles')
        ax.legend()
        
        ## Add plane visualization (optional)
        #xx1, yy1 = np.meshgrid(np.linspace(-3e5, 2e5, 10), np.linspace(-1.5e5, 0.5e5, 10))
        #zz1 = (-normal1[0]*(xx1 - centroid1[0]) - normal1[1]*(yy1 - centroid1[1])) / normal1[2] + centroid1[2]
        #ax.plot_surface(xx1, yy1, zz1, alpha=0.1, color='blue')
        #
        #xx2, yy2 = np.meshgrid(np.linspace(-1.5e5, 0.5e5, 10), np.linspace(-4e4, 3e4, 10))
        #zz2 = (-normal2[0]*(xx2 - centroid2[0]) - normal2[1]*(yy2 - centroid2[1])) / normal2[2] + centroid2[2]
        #ax.plot_surface(xx2, yy2, zz2, alpha=0.1, color='orange')
      
        ## Set equal aspect ratio (important for proper visualization)
        #max_range = np.array([circle1_points.max()-circle1_points.min(), 
        #                      circle2_points.max()-circle2_points.min()]).max() * 0.5
        #mid_x = (circle1_points[:,0].mean() + circle2_points[:,0].mean()) * 0.5
        #mid_y = (circle1_points[:,1].mean() + circle2_points[:,1].mean()) * 0.5
        #mid_z = (circle1_points[:,2].mean() + circle2_points[:,2].mean()) * 0.5
        #ax.set_xlim(mid_x - max_range, mid_x + max_range)
        #ax.set_ylim(mid_y - max_range, mid_y + max_range)
        #ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.show()

        exit()









        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d') 

        #ax.plot(xs[:,0], xs[:,1], xs[:,2], color="orange", alpha=0.6, linewidth = 2.0)
        #ax.plot(A1[0], A1[1], A1[2], color="black", alpha=0.6, linewidth = 2.0)

        ## Adjusted from Truth
        #ax.plot(B1[0], B1[1], B1[2], color="purple", alpha=0.6, linewidth = 2.0)
        #ax.plot(B2[0], B2[1], B2[2], color="purple", alpha=0.6, linewidth = 2.0)

        ## Fixed point
        #ax.plot(_A1[0], _A1[1], _A1[2], color="orange", alpha=0.6, linestyle = ":", linewidth = 2.0)
        #ax.plot(_A2[0], _A2[1], _A2[2], color="orange", alpha=0.6, linestyle = "--", linewidth = 2.0)

        ## Adjusted fixed point
        #ax.plot(_B1[0], _B1[1], _B1[2], color="purple", alpha=0.6, linestyle = ":", linewidth = 2.0)
        #ax.plot(_B2[0], _B2[1], _B2[2], color="purple", alpha=0.6, linestyle = "--", linewidth = 2.0)

        ##ax.plot(Cx[0], Cx[1], Cx[2], color="red", alpha=0.6)
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        #ax.set_zlabel('Z')
        #plt.show()


        #exit()
        print(self.sol1.H)
        print(self.sol2.H)






        exit()
        #plane_met = np.array([x0, y0, z0])
        #nrm_met = plane_met / np.linalg.norm(plane_met)
        #du = np.array([plane_met[1], -plane_met[0], 0])
        #du = du / np.linalg.norm(du)
        #dv = np.cross(plane_met, du)
        #dv = dv / np.linalg.norm(dv)
        #rho = (x0**2 + y0**2 + z0**2)**0.5





        try: N, N_ = self.sol1.N, self.sol2.N; self.failed = False
        except np.linalg.LinAlgError: self.failed = True
        if self.failed: return 

        n_   = self.S.T.dot(N_).dot(self.S)
        n    = self.S.T.dot(N ).dot(self.S)
#        plotthis([self.sol1.H, self.sol2.H], "figs/yx")
#        plotthis([self.sol1.H_projx, self.sol2.H_projx], "figs/yx")
#        plotthis([self.sol1.H_projy, self.sol2.H_projy], "figs/yy")
#        plotthis([self.sol1.H_perp , self.sol2.H_perp] , "figs/yx")

        try: v, l, q22 = intersections_ellipses(N , n_)
        except ValueError: self.failed = True
#        print(q22, d, v)
        print(v)

        v_   = [self.S.dot(sol) for sol in v]
        if not len(v_): self.failed = True
        if self.failed: return 
        
        plx = []
        cx = np.cross(self.sol1.H, self.sol2.H)
        cx = cx / np.linalg.norm(cx)
        h1x = self.sol1.H / np.linalg.norm(self.sol1.H)
        h2x = self.sol2.H / np.linalg.norm(self.sol2.H)

        plx.append(cx)
        plx.append(h1x)
        plx.append(h2x)

        rl = cx.dot(V0)
        rl *= sum([rl[i][2]**2 for i in range(3)])**-0.5
        self.met_con = (math.atan2(rl[1][2], rl[0][2])*(180/math.pi), math.asin(rl[2][2])*(180/math.pi))

        K, K_ = [ss.H.dot(np.linalg.inv(ss.H_perp)) for ss in self.solutionSets]
        nu1 = np.array([K.dot(s)   for s  in v ])
        nu2 = np.array([K_.dot(s_) for s_ in v_])
        rl = rl[:,2]*np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        ang = np.cross(nu1[0], nu2[0]).dot(cx)
        ang *= sum(ang**2)**-0.5

        self.nu_con = (math.atan2(ang[1], ang[0])*(180/math.pi), math.asin(ang[2])*(180/math.pi))
#        print(self.nu_con)
        plx.append(rl)
#        plotthis(plx, "figs/data-" + itx)
        #print(nu1)
        #exit()
        self.failed = True
        dx = nu1+nu2
#        print("-----")
#        for i in range(len(dx)):
#            vx = dx[i]
#            dl = vx * sum(vx**2)**-0.5
#            dl = np.dot(dl, cx)
#            dl *= sum(dl**2)**-0.5
#            print(math.atan2(dl[1], dl[0])*(180/math.pi), math.asin(dl[2])*(180/math.pi))
        #exit()

        #if not v_:
        #    es = [ss.H_perp for ss in self.solutionSets]
        #    met = self.ev
        #    def nus(ts): return tuple(e.dot([math.cos(t), math.sin(t), 1]) for e, t in zip(es, ts))
        #    def residuals(params): return sum(nus(params), -met)[:2]
        #    ts, _ = leastsq(residuals, [0, 0], ftol=5e-5, epsfcn=0.01)
        #    self.lsq = True
        #    if sum(residuals(ts)**2) > 1e-6: pass
        #    else: v, v_ = [[i] for i in nus(ts)]
        for k, v in {"perp" : v , "perp_":  v_, "n_" : n_, "n" : n}.items(): setattr(self, k, v)
        self.failed = len(v_) == 0

    def nunu_s(self):
        """Solution pairs for neutrino momenta"""
        out = {0 : [], 1 : []}
        if self.failed: return out
        pairs = []
        for s, s_ in zip(self.perp, self.perp_): pairs.append((np.dot(s.T , self.n_).dot(s) - np.dot(s_.T, self.n).dot(s_))**2)
        K, K_ = [ss.H.dot(np.linalg.inv(ss.H_perp)) for ss in self.solutionSets]
        nu1 = np.array([K.dot(s)   for s  in self.perp ])
        nu2 = np.array([K_.dot(s_) for s_ in self.perp_])
        for i in range(len(nu1)):
            x = pairs[i]
            p1 = Neutrino()
            p1.distance = x
            p1.px = nu1[i][0]
            p1.py = nu1[i][1]
            p1.pz = nu1[i][2]
            out[0].append(p1)

            p2 = Neutrino()
            p2.distance = x
            p2.px = nu2[i][0]
            p2.py = nu2[i][1]
            p2.pz = nu2[i][2]
            out[1].append(p2)
        return out
