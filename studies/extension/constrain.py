import math
import numpy as np
from math import sqrt, cos, sin, atan2, pi
from common import get_mw, get_mt2, intersections_ellipses, UnitCircle, NuSol
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import vector


# ================== VISUALIZATION ==================
def plot_solution(params, constants):
    mW1, mT1, mW2, mT2, t1, t2 = params
    sol1, sol2 = constants
    
    # Create trajectories
    t_vals = np.linspace(0, 2*np.pi, 100)
    traj1 = np.array([sol1.H @ [np.cos(t), np.sin(t), 1] for t in t_vals])
    traj2 = np.array([sol2.H @ [np.cos(t), np.sin(t), 1] for t in t_vals])
    
    # Compute optimized points
    point1 = sol1.H.dot([np.cos(t1), np.sin(t1), 1])
    point2 = sol2.H.dot([np.cos(t2), np.sin(t2), 1])
    distance = np.linalg.norm(point1 - point2)
    
    # Create plot
    fig = plt.figure(figsize=(15, 10))
    
    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(traj1[:, 0], traj1[:, 1], traj1[:, 2], 'b-', alpha=0.3, label='ν1 Trajectory')
    ax1.plot(traj2[:, 0], traj2[:, 1], traj2[:, 2], 'r-', alpha=0.3, label='ν2 Trajectory')
    ax1.plot([point1[0], point2[0]], 
             [point1[1], point2[1]], 
             [point1[2], point2[2]], 'k-', linewidth=2, label=f'Distance: {distance:.2f} GeV')
    ax1.scatter(*point1, c='blue', s=100, label='Opt ν1')
    ax1.scatter(*point2, c='red', s=100, label='Opt ν2')
    ax1.set_xlabel('Px (GeV)')
    ax1.set_ylabel('Py (GeV)')
    ax1.set_zlabel('Pz (GeV)')
    ax1.set_title('3D Neutrino Momentum Space')
    ax1.legend()
    ax1.grid(True)
    
    # 2D Projection
    ax2 = fig.add_subplot(122)
    ax2.plot(traj1[:, 0], traj1[:, 1], 'b-', alpha=0.3, label='ν1 Trajectory')
    ax2.plot(traj2[:, 0], traj2[:, 1], 'r-', alpha=0.3, label='ν2 Trajectory')
    ax2.plot([point1[0], point2[0]], [point1[1], point2[1]], 'k-', linewidth=2)
    ax2.scatter(*point1[:2], c='blue', s=100, label='Opt ν1')
    ax2.scatter(*point2[:2], c='red', s=100, label='Opt ν2')
    ax2.set_xlabel('Px (GeV)')
    ax2.set_ylabel('Py (GeV)')
    ax2.set_title('Transverse Plane Projection')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(f"Closest Points: Distance = {distance:.2f} GeV", fontsize=16)
    plt.tight_layout()
    plt.show()

class nusol(NuSol):
    def __init__(self, b, mu, mW, mT):
        NuSol.__init__(self, b, mu, mW**2, mT**2)
        
    @property
    def OptimizeMW(self): return nusol(self.b, self.mu, max(get_mw(self))**2, self.mT2)
    @property
    def OptimizeMT(self): return get_mt2(self)
    
    def dx0_dmW(self):    return -math.sqrt(self.mW2) / self.mu.e
    def dx0p_dmW(self):   return math.sqrt(self.mW2) / self.b.e
    def dx0p_dmT(self):   return -math.sqrt(self.mT2) / self.b.e
    def dSx_dmW(self):    return (self.dx0p_dmW() * self.mu.beta) / self.mu.beta**2
    def dSx_dmT(self):    return (self.dx0p_dmT() * self.mu.beta) / self.mu.beta**2
    def dSy_dmT(self):    return (-self.c * self.dSx_dmT()) / self.s
    def dx1_dmW(self):    return self.dSx_dmW() - (self.dSx_dmW() + self.w * self.dSy_dmW()) / self.Om2
    def dSy_dmW(self):    return (self.dx0_dmW() / self.b.beta - self.c * self.dSx_dmW()) / self.s
    def dy1_dmW(self):
        dSx_dmW = self.dSx_dmW()
        dSy_dmW = self.dSy_dmW()
        numerator = dSx_dmW + self.w * dSy_dmW
        return dSy_dmW - self.w * numerator / self.Om2
    
    def dx1_dmT(self):
        dSx_dmT = self.dSx_dmT()
        dSy_dmT = self.dSy_dmT()
        numerator = dSx_dmT + self.w * dSy_dmT
        return dSx_dmT - numerator / self.Om2
    
    def dy1_dmT(self):
        dSx_dmT = self.dSx_dmT()
        dSy_dmT = self.dSy_dmT()
        numerator = dSx_dmT + self.w * dSy_dmT
        return dSy_dmT - self.w * numerator / self.Om2
    
    def dZ2_dmW(self):
        dx1_dmW = self.dx1_dmW()
        dSx_dmW = self.dSx_dmW()
        dSy_dmW = self.dSy_dmW()
        dx0_dmW = self.dx0_dmW()
        
        term1 =  2 * self.Om2 * self.x1 * dx1_dmW
        term2 = -2 * (self.Sy - self.w * self.Sx) * (dSy_dmW - self.w * dSx_dmW)
        term3 =  2 * self.x0 * dx0_dmW - 2 * math.sqrt(self.mW2)
        return term1 + term2 + term3
    
    def dZ2_dmT(self):
        dx1_dmT = self.dx1_dmT()
        dSx_dmT = self.dSx_dmT()
        dSy_dmT = self.dSy_dmT()
        
        term1 =  2 * self.Om2 * self.x1 * dx1_dmT
        term2 = -2 * (self.Sy - self.w * self.Sx) * (dSy_dmT - self.w * dSx_dmT)
        return term1 + term2
    
    def dZ_dmW(self): return (0.5 / self.Z) * self.dZ2_dmW()
    def dZ_dmT(self): return (0.5 / self.Z) * self.dZ2_dmT()
    
    def dB_dmW(self):
        sqrt_Om2 = sqrt(self.Om2)
        dZ_dmW = self.dZ_dmW()
        dx1_dmW = self.dx1_dmW()
        dy1_dmW = self.dy1_dmW()
        
        da = dZ_dmW / sqrt_Om2
        db = self.w * da
        dc = dZ_dmW
        dd = dx1_dmW
        de = dy1_dmW
        
        return np.array([[da, 0, dd], [db, 0, de], [0, dc, 0]])
    
    def dB_dmT(self):
        sqrt_Om2 = sqrt(self.Om2) 
        dZ_dmT = self.dZ_dmT()
        dx1_dmT = self.dx1_dmT()
        dy1_dmT = self.dy1_dmT()
        
        da = dZ_dmT / sqrt_Om2
        db = self.w * da
        dc = dZ_dmT
        dd = dx1_dmT
        de = dy1_dmT
        
        return np.array([[da, 0, dd], [db, 0, de], [0, dc, 0]])
    
    # Second derivatives
    def d2x0_dmW2(self):  return -1 / self.mu.e
    def d2x0p_dmW2(self): return  1 / self.b.e
    def d2x0p_dmT2(self): return -1 / self.b.e
    def d2Sx_dmW2(self):  return (self.d2x0p_dmW2() * self.mu.beta) / self.mu.beta**2
    def d2Sy_dmW2(self):  return (self.d2x0_dmW2() / self.b.beta - self.c * self.d2Sx_dmW2()) / self.s
    def d2Sx_dmT2(self):  return (self.d2x0p_dmT2() * self.mu.beta) / self.mu.beta**2
    def d2Sy_dmT2(self):  return (-self.c * self.d2Sx_dmT2()) / self.s
    
    def d2x1_dmW2(self):
        d2Sx_dmW2 = self.d2Sx_dmW2()
        d2Sy_dmW2 = self.d2Sy_dmW2()
        numerator = d2Sx_dmW2 + self.w * d2Sy_dmW2
        return d2Sx_dmW2 - numerator / self.Om2
    
    def d2y1_dmW2(self):
        d2Sx_dmW2 = self.d2Sx_dmW2()
        d2Sy_dmW2 = self.d2Sy_dmW2()
        numerator = d2Sx_dmW2 + self.w * d2Sy_dmW2
        return d2Sy_dmW2 - self.w * numerator / self.Om2
    
    def d2x1_dmT2(self):
        d2Sx_dmT2 = self.d2Sx_dmT2()
        d2Sy_dmT2 = self.d2Sy_dmT2()
        numerator = d2Sx_dmT2 + self.w * d2Sy_dmT2
        return d2Sx_dmT2 - numerator / self.Om2
    
    def d2y1_dmT2(self):
        d2Sx_dmT2 = self.d2Sx_dmT2()
        d2Sy_dmT2 = self.d2Sy_dmT2()
        numerator = d2Sx_dmT2 + self.w * d2Sy_dmT2
        return d2Sy_dmT2 - self.w * numerator / self.Om2
    
    def d2Z2_dmW2(self):
        dx1_dmW = self.dx1_dmW()
        dSx_dmW = self.dSx_dmW()
        dSy_dmW = self.dSy_dmW()
        dx0_dmW = self.dx0_dmW()
        
        d2x0_dmW2 = self.d2x0_dmW2()
        d2Sx_dmW2 = self.d2Sx_dmW2()
        d2Sy_dmW2 = self.d2Sy_dmW2()
        d2x1_dmW2 = self.d2x1_dmW2()
        
        S_term = self.Sy - self.w * self.Sx
        dS_term_dmW = dSy_dmW - self.w * dSx_dmW
        d2S_term_dmW2 = d2Sy_dmW2 - self.w * d2Sx_dmW2
        
        term1 =  2 * self.Om2 * (dx1_dmW**2 + self.x1 * d2x1_dmW2)
        term2 = -2 * (dS_term_dmW**2 + S_term * d2S_term_dmW2)
        term3 =  2 * (dx0_dmW**2 + self.x0 * d2x0_dmW2) - 2
        return term1 + term2 + term3
    
    def d2Z2_dmT2(self):
        dS_dmT = self.dSy_dmT() - self.w * self.dSx_dmT()
        d2S_dmT2 = self.d2Sy_dmT2() - self.w * self.d2Sx_dmT2()
        
        term1 =  2 * self.Om2 * (self.dx1_dmT()**2 + self.x1 * self.d2x1_dmT2())
        term2 = -2 * (dS_dmT**2 + (self.Sy - self.w * self.Sx) * d2S_dmT2)
        
        return term1 + term2
    
    def d2Z2_dmWdmT(self):
        dx1_dmW = self.dx1_dmW()
        dx1_dmT = self.dx1_dmT()
        dSx_dmW = self.dSx_dmW()
        dSx_dmT = self.dSx_dmT()
        dSy_dmW = self.dSy_dmW()
        dSy_dmT = self.dSy_dmT()
        
        d2Sx_dmWdmT = 0
        d2Sy_dmWdmT = 0
        d2x1_dmWdmT = 0
         
        S_term = self.Sy - self.w * self.Sx
        dS_term_dmW = dSy_dmW - self.w * dSx_dmW
        dS_term_dmT = dSy_dmT - self.w * dSx_dmT
        d2S_term_dmWdmT = d2Sy_dmWdmT - self.w * d2Sx_dmWdmT

        term1 =  2 * self.Om2 * (dx1_dmW * dx1_dmT + self.x1 * d2x1_dmWdmT)
        term2 = -2 * (dS_term_dmW * dS_term_dmT + S_term * d2S_term_dmWdmT)
        return term1 + term2
    
    def d2Z_dmW2(self):   return (0.5 * self.d2Z2_dmW2() / self.Z - 0.25 * self.dZ2_dmW()**2 / self.Z**3)
    def d2Z_dmT2(self):   return (0.5 * self.d2Z2_dmT2() / self.Z - 0.25 * self.dZ2_dmT()**2 / self.Z**3)
    def d2Z_dmWdmT(self): return (0.5 * self.d2Z2_dmWdmT() / self.Z - 0.25 * self.dZ2_dmW() * self.dZ2_dmT() / self.Z**3)
    
    def d2a_dmW2(self):
        sqrt_Om2 = sqrt(self.Om2)
        return (self.d2Z_dmW2()* sqrt_Om2 - self.dZ_dmW()**2 / sqrt_Om2) / self.Om2
 
    def d2a_dmT2(self):
        sqrt_Om2 = sqrt(self.Om2)
        return (self.d2Z_dmT2() * sqrt_Om2 - self.dZ_dmT()**2 / sqrt_Om2) / self.Om2
    
    def d2b_dmT2(self): return self.w * self.d2a_dmT2()
    def d2c_dmT2(self): return self.d2Z_dmT2()
    def d2d_dmT2(self): return self.d2x1_dmT2()
    def d2e_dmT2(self): return self.d2y1_dmT2()

    def d2b_dmW2(self): return self.w * self.d2a_dmW2()
    def d2c_dmW2(self): return self.d2Z_dmW2()
    def d2d_dmW2(self): return self.d2x1_dmW2()
    def d2e_dmW2(self): return self.d2y1_dmW2()
 


    def d2B_dmW2(self):
        d2a = self.d2a_dmW2()
        d2b = self.d2b_dmW2()
        d2c = self.d2c_dmW2()
        d2d = self.d2d_dmW2()
        d2e = self.d2e_dmW2()
        return np.array([[d2a, 0, d2d], [d2b, 0, d2e], [0, d2c, 0]])
    
    def d2B_dmT2(self):
        d2a = self.d2a_dmT2()
        d2b = self.d2b_dmT2()
        d2c = self.d2c_dmT2()
        d2d = self.d2d_dmT2()
        d2e = self.d2e_dmT2()
        return np.array([[d2a, 0, d2d], [d2b, 0, d2e], [0, d2c, 0]])
    
    def d2B_dmWdmT(self): return np.zeros((3, 3))

def jacobian_and_hessian(params, b1, mu1, b2, mu2):
    mW1, mT1, mW2, mT2, t1, t2 = params
    
    # Create NuSol instances
    sol1 = nusol(b1, mu1, mW1, mT1)
    sol2 = nusol(b2, mu2, mW2, mT2)
    
    # Compute base matrices and rotation matrices
    B1, B2 = sol1._BaseMatrix, sol2._BaseMatrix
    R_T1, R_T2 = sol1.R_T, sol2.R_T
    
    # Define vectors
    v1 = np.array([cos(t1), sin(t1), 1])
    v2 = np.array([cos(t2), sin(t2), 1])
    
    # Compute H vectors
    d = (sol1.H.dot(v1),  sol2.H.dot(v2))
    r =  sol1.H.dot(v1) - sol2.H.dot(v2)
  
    # First derivatives of v vectors
    dv1_dt1 = np.array([-sin(t1), cos(t1), 0])
    dv2_dt2 = np.array([-sin(t2), cos(t2), 0])
    
    # Second derivatives of v vectors
    d2v1_dt1 = np.array([-cos(t1), -sin(t1), 0])
    d2v2_dt2 = np.array([-cos(t2), -sin(t2), 0])
    
    # =====================
    # Jacobian Calculation
    # =====================
    # Event 1 derivatives
    dB1_dmW1 = sol1.dB_dmW()
    dB1_dmT1 = sol1.dB_dmT()
    dH1_dmW1 = R_T1.dot(dB1_dmW1).dot(v1)
    dH1_dmT1 = R_T1.dot(dB1_dmT1).dot(v1)

    dH1_dt1  = R_T1.dot(B1).dot(dv1_dt1)
    dH2_dt2  = R_T2.dot(B2).dot(dv2_dt2)
    
    # Event 2 derivatives
    dB2_dmW2 = sol2.dB_dmW()
    dB2_dmT2 = sol2.dB_dmT()
    dH2_dmW2 = R_T2.dot(dB2_dmW2).dot(v2)
    dH2_dmT2 = R_T2.dot(dB2_dmT2).dot(v2)
    
    # Jacobian components
    J = np.zeros(6)
    J[0] =  2 * np.dot(r, dH1_dmW1)
    J[1] =  2 * np.dot(r, dH1_dmT1)
    J[2] = -2 * np.dot(r, dH2_dmW2)
    J[3] = -2 * np.dot(r, dH2_dmT2)
    J[4] =  2 * np.dot(r, dH1_dt1)
    J[5] = -2 * np.dot(r, dH2_dt2)
    
    # =====================
    # Hessian Calculation
    # =====================
    H = np.zeros((6, 6))
    
    # Compute all second derivatives
    # Event 1
    d2B1_dmW12    = sol1.d2B_dmW2()
    d2B1_dmT12    = sol1.d2B_dmT2()
    d2B1_dmW1dmT1 = sol1.d2B_dmWdmT()
    
    d2H1_dmW12    = R_T1.dot(d2B1_dmW12).dot(v1)
    d2H1_dmT12    = R_T1.dot(d2B1_dmT12).dot(v1)
    d2H1_dt12     = R_T1.dot(B1).dot(d2v1_dt1)
    d2H1_dmW1dmT1 = R_T1.dot(d2B1_dmW1dmT1).dot(v1)
    d2H1_dmW1dt1  = R_T1.dot(dB1_dmW1).dot(dv1_dt1)
    d2H1_dmT1dt1  = R_T1.dot(dB1_dmT1).dot(dv1_dt1)
    
    # Event 2
    d2B2_dmW22    = sol2.d2B_dmW2()
    d2B2_dmT22    = sol2.d2B_dmT2()
    d2B2_dmW2dmT2 = sol2.d2B_dmWdmT()
    
    d2H2_dmW22    = R_T2.dot(d2B2_dmW22).dot(v2)
    d2H2_dmT22    = R_T2.dot(d2B2_dmT22).dot(v2)
    d2H2_dt22     = R_T2.dot(B2).dot(d2v2_dt2)
    d2H2_dmW2dmT2 = R_T2.dot(d2B2_dmW2dmT2).dot(v2)
    d2H2_dmW2dt2  = R_T2.dot(dB2_dmW2).dot(dv2_dt2)
    d2H2_dmT2dt2  = R_T2.dot(dB2_dmT2).dot(dv2_dt2)
    grads = [dH1_dmW1, dH1_dmT1, -dH2_dmW2, -dH2_dmT2, dH1_dt1, -dH2_dt2]
    
    for i in range(6):
        for j in range(6):
            H[i, j] = 2 * np.dot(grads[i], grads[j])
    
    # Add second derivative terms
    # Event 1 terms
    H[0, 0] += 2 * np.dot(r, d2H1_dmW12)
    H[1, 1] += 2 * np.dot(r, d2H1_dmT12)
    H[4, 4] += 2 * np.dot(r, d2H1_dt12)
    
    H[0, 1] += 2 * np.dot(r, d2H1_dmW1dmT1)
    H[1, 0] += 2 * np.dot(r, d2H1_dmW1dmT1)
    
    H[0, 4] += 2 * np.dot(r, d2H1_dmW1dt1)
    H[4, 0] += 2 * np.dot(r, d2H1_dmW1dt1)
    
    H[1, 4] += 2 * np.dot(r, d2H1_dmT1dt1)
    H[4, 1] += 2 * np.dot(r, d2H1_dmT1dt1)
    
    # Event 2 terms (note sign for H2)
    H[2, 2] += 2 * np.dot(r, -d2H2_dmW22)
    H[3, 3] += 2 * np.dot(r, -d2H2_dmT22)
    H[5, 5] += 2 * np.dot(r, -d2H2_dt22)
    
    H[2, 3] += 2 * np.dot(r, -d2H2_dmW2dmT2)
    H[3, 2] += 2 * np.dot(r, -d2H2_dmW2dmT2)
    
    H[2, 5] += 2 * np.dot(r, -d2H2_dmW2dt2)
    H[5, 2] += 2 * np.dot(r, -d2H2_dmW2dt2)
    
    H[3, 5] += 2 * np.dot(r, -d2H2_dmT2dt2)
    H[5, 3] += 2 * np.dot(r, -d2H2_dmT2dt2)
    
    # Cross-event angle terms
    d2H_dt1dt2 = np.zeros(3)
    H[4, 5] += 2 * np.dot(r, d2H_dt1dt2)
    H[5, 4] += 2 * np.dot(r, d2H_dt1dt2)
    
    return d, J, H

def objective(masses, const):
    sol1, sol2 = const
    sol1 = nusol(sol1.b, sol1.mu, masses[0], masses[1])
    sol2 = nusol(sol2.b, sol2.mu, masses[2], masses[3])
    #sol1, sol2 = sol1.OptimizeMT, sol2.OptimizeMT
    const[0] = sol1; const[1] = sol2

    mass = [sqrt(i) for i in [sol1.mW2, sol1.mT2, sol2.mW2, sol2.mT2]] + [masses[4], masses[5]]
    d, jaco, hess = jacobian_and_hessian(masses, sol1.b, sol1.mu, sol2.b, sol2.mu)

    met = np.array([106.435841, -141.293331, 100])
    S = np.outer([met[0], met[1], met[2]], [0, 0, 1]) - UnitCircle() 
    v, l, q22 = intersections_ellipses(sol1.N, S.T.dot(sol2.N).dot(S))

    data = []
    v_ = [S.dot(sol) for sol in v]
    for i,j in zip(v, v_):
        h1, h2 = np.linalg.inv(sol1.H).dot(i), np.linalg.inv(sol2.H).dot(j)
        t1, t2 = np.arctan2(h1[1], h1[0]), np.atan2(h2[1], h2[0])
        data += [sol1.dist2(sol2, t1, t2)]
  
    #d = (met - (d[0] + d[1]))**2
    print("+>", [round(i, 2) for i in np.array(masses).tolist()], "|", d[0], len(data), jaco)
    if not len(data): return d, jaco, hess
    idx = data.index(min(data))
    K, K_ = [ss.H.dot(np.linalg.inv(ss.H_perp)) for ss in [sol1, sol2]]
    nu1   = [K.dot(s)   for s  in v ][idx]
    nu2   = [K_.dot(s_) for s_ in v_][idx]

    tp1 = (sol1.b + sol1.mu + vector.obj(px = nu1[0], py = nu1[1], pz = nu1[2], E = sum(nu1**2)**0.5))
    tp2 = (sol2.b + sol2.mu + vector.obj(px = nu2[0], py = nu2[1], pz = nu2[2], E = sum(nu2**2)**0.5))

    wp1 = (sol1.mu + vector.obj(px = nu1[0], py = nu1[1], pz = nu1[2], E = sum(nu1**2)**0.5))
    wp2 = (sol2.mu + vector.obj(px = nu2[0], py = nu2[1], pz = nu2[2], E = sum(nu2**2)**0.5))
    print("->", tp1.tau, tp2.tau, wp1.tau, wp2.tau)
    return d, jaco, hess

def levenberg_marquardt(x0, const,lambda0=1e-3, max_iter=100, tol=1e-6, min_lambda=1e-10, max_lambda=1e10):
    lambda_current = lambda0
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        f_x, g, H = objective(x, const)
        grad_norm = np.linalg.norm(g)
        if grad_norm < tol: print(f"Converged at iteration {i}: grad_norm={grad_norm:.3e}"); break

        D = np.diag(np.maximum(np.abs(np.diag(H)), 1e-8))  # Prevent zero diagonals
        try: delta = np.linalg.solve(H + lambda_current * D, -g)
        except np.linalg.LinAlgError: lambda_current *= 10; continue

        # Evaluate new point
        x_new = x + delta
        f_new = objective(x_new.tolist(), const)[0]
        f_new, f_x = sum(f_new), sum(f_x) 

        if f_new < f_x:
            lambda_current = max(lambda_current / 10, min_lambda)
            x = x_new
#            print(f"Iter {i}: f={f_x:.6e} → {f_new:.6e}, "f"|g|={grad_norm:.3e}, λ={lambda_current:.1e}")
        else: 
            lambda_current = min(lambda_current * 10, max_lambda)
#            print(f"Iter {i}: Rejected (f {f_new:.6e} > {f_x:.6e}), " f"λ={lambda_current:.1e}")
        if not (i+1) % 100: plot_solution(x, const)
    return x


# ================== MAIN EXECUTION ==================
if __name__ == "__main__":
    b1  = np.array([-19.766428, -40.022249 ,   69.855886, 83.191328 ])
    b2  = np.array([107.795878, -185.326183,  -67.989162, 225.794953])
    mu1 = np.array([-14.306453, -47.019613 ,    3.816470, 49.295996 ])
    mu2 = np.array([4.941336  , -104.097506, -103.640669, 146.976547])
    
    initial_params = np.array([80.385, 172.62, 80.385, 172.62, 0.0, 0.0])
    sol1 = nusol(b1, mu1, initial_params[0], initial_params[1])
    sol2 = nusol(b2, mu2, initial_params[2], initial_params[3])
    print(sol1.Z2, sol2.Z2)
    exit()

    optimized_masses = levenberg_marquardt(
            initial_params, [sol1, sol2], 
            lambda0=1e-9, max_iter=100000, tol=1e-12, 
            min_lambda=1e-12, max_lambda=1e9
    )
