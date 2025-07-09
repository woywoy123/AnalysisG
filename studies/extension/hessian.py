import vector
import math
import numpy as np
from scipy.optimize import minimize
from common import NuSol, UnitCircle, intersections_ellipses, figs, plot_ellipse, plot, plot_plane, _make_ellipse, find_intersection_points

def analytic_neutrino_momentum(b, mu, mT, mW): return NuSol(b, mu, mW**2, mT**2)

class LorentzVector:
    def __init__(self, px, py, pz, E = None):
        self.px = px
        self.py = py
        self.pz = pz
        self.vec = np.array([px, py, pz])
        if E is None: E = (self.p**2)**0.5
        self.E = E
    
    @property
    def p(self):
        return np.linalg.norm(self.vec)
    
    @property
    def beta(self):
        return self.p / self.E if self.E > 0 else 0
    
    @property
    def gamma(self):
        return self.E / self.mass if self.mass > 0 else 1
    
    @property
    def mass(self):
        return math.sqrt(max(0, self.E**2 - self.p**2))

    def __add__(self, other):
        return LorentzVector(
                self.px + other.px, 
                self.py + other.py, 
                self.pz + other.pz, 
                self.E  + other.E)

def objective(masses):
    mt1, mw1, mt2, mw2 = masses

    met = np.array([106.435841, -141.293331, 0])
    b1  = np.array([-19.766428, -40.022249 ,   69.855886, 83.191328 ])
    b2  = np.array([107.795878, -185.326183,  -67.989162, 225.794953])
    mu1 = np.array([-14.306453, -47.019613 ,    3.816470, 49.295996 ])
    mu2 = np.array([4.941336  , -104.097506, -103.640669, 146.976547])
    nu1T = np.array([51.19149113841701,  -8.428439462306061, -7.283688512367609])
    nu2T = np.array([78.7694069553391 , -55.373725298361485, -57.24324147281612])
    #met = (nu1T+nu2T)
    ms = [142.703747, 76.765395, 164.548815, 93.568769]


    #_alpha = np.radians(0)
    #_gamma = np.radians(0)
    #_beta  = np.radians(0)

    #ca, sa = np.cos(_alpha), np.sin(_alpha)
    #cg, sg = np.cos(_gamma), np.sin(_gamma)
    #cb, sb = np.cos(_beta) , np.sin(_beta)
    
    #R = np.array([
    #    [cb*cg, sa*sb*cg - ca*sg, ca*sb*cg + sa*sg], 
    #    [cb*sg, sa*sb*sg + ca*cg, ca*sb*sg - sa*cg], 
    #    [-sb  , sa*cb           , ca*cb           ]
    #])

  #  S = R.dot(
    S = np.outer([met[0], met[1], met[2]], [0, 0, 1]) - UnitCircle() #)

    sol1 = analytic_neutrino_momentum(b1, mu1, mt1, mw1).OptimizeMT
    sol2 = analytic_neutrino_momentum(b2, mu2, mt2, mw2).OptimizeMT

    dist = (sol1.ellipse_property["centroid"] - sol2.ellipse_property["centroid"])
    nr1, nr2 = sol1.ellipse_property["normal"], sol2.ellipse_property["normal"]

    #_sol1 = analytic_neutrino_momentum(b1, mu1, ms[0], ms[1])
    #_sol2 = analytic_neutrino_momentum(b2, mu2, ms[2], ms[3])

    #ax = figs()
    #plot_ellipse(sol1.H , ax, "Ellipse 1", "red" , "-.")
    #plot_ellipse(sol2.H , ax, "Ellipse 2", "blue", "-.")

    #plot_ellipse(_sol1.H, ax, "Ellipse 1", "black" , "-")
    #plot_ellipse(_sol2.H, ax, "Ellipse 2", "purple", "-")
    #plot()

    f1, n1, p1 = sol1.H, sol1.N, sol1.H_perp
    f2, n2, p2 = sol2.H, sol2.N, sol2.H_perp
    dtx1 = np.array([np.cos(sol1.angle_x), np.sin(sol1.angle_y), 1])*np.cos(sol1.angle_z)
    dtx2 = np.array([np.cos(sol2.angle_x), np.sin(sol2.angle_y), 1])*np.cos(sol2.angle_z)



    th1, th2 = None, None
    v, v_, d = [], [], []
    angles = sol1.get_intersection_angle(p1, p2)
    for sol in angles:
        #v.append( np.array(_make_ellipse(f1, sol[0])))
        #v_.append(np.array(_make_ellipse(f2, sol[1])))
        d.append(sol1.dist2(sol2, sol[0], sol[1]))

    th1, th2 = angles[d.index(min(d))] if len(angles) else (0, 0)
    dist = np.array([sol1.dist2(sol2, th1, th2)])
    dz = np.array(list(sol1.dD_dM(sol2, th1, th2)))
    print(dz, th1, th2)

    fv, l, q22 = intersections_ellipses(n1, S.T.dot(n2).dot(S))
    fv_ = [S.dot(sol) for sol in fv]
    v += fv; v_ += fv_
    data = []
    for i,j in zip(v, v_):
        try: h1, h2 = np.linalg.inv(f1).dot(i), np.linalg.inv(f2).dot(j)
        except np.linalg.LinAlgError: continue
        t1, t2 = np.arctan2(h1[1], h1[0]), np.atan2(h2[1], h2[0])
        h1, h2 = _make_ellipse(f1, t1)   , _make_ellipse(f2, t2)
        data += [sum((h1 - h2)**2)**0.5]

    hess = None
    print(masses, np.array(data), dist, q22)
    if not len(data): return dist, dz, hess 

    idx = data.index(min(data))
    K, K_ = [ss.H.dot(np.linalg.inv(ss.H_perp)) for ss in [sol1, sol2]]
    nu1   = [K.dot(s)   for s  in v ][idx]
    nu2   = [K_.dot(s_) for s_ in v_][idx]


    tT1 = LorentzVector(*b1) + LorentzVector(*mu1) + LorentzVector(*nu1T)
    tt1 = LorentzVector(*b1) + LorentzVector(*mu1) + LorentzVector(*nu1)
    tT2 = LorentzVector(*b2) + LorentzVector(*mu2) + LorentzVector(*nu2T)
    tt2 = LorentzVector(*b2) + LorentzVector(*mu2) + LorentzVector(*nu2)
    chi2 = ((tT1.mass - tt1.mass)**2 + (tT2.mass - tt2.mass)**2)
    print("->", chi2, tt1.mass, tt2.mass)
    #ax = figs()
    #plot_ellipse(S, ax, "Ellipse 1", "red")
    #plot_ellipse(sol2.H, ax, "Ellipse 2", "blue")
    #plot_plane(sol1.H_perp, sol2.H_perp, None, None, ax) #v[0],np.array(l[0]), ax) 
    #plot()

    p_nu_x = (met[0] + (b1[0] + b2[0] + mu1[0] + mu2[0] + nu1[0] + nu2[0]))
    p_nu_y = (met[1] + (b1[1] + b2[1] + mu1[1] + mu2[1] + nu1[1] + nu2[1]))
    p_nu_z = (met[2] + (b1[2] + b2[2] + mu1[2] + mu2[2] + nu1[2] + nu2[2]))
    return dist, dz, hess 


def levenberg_marquardt(x0, lambda0=1e-3, max_iter=100, tol=1e-6, min_lambda=1e-10, max_lambda=1e10):
    lambda_current = lambda0
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        f_x, g, H = objective(x.tolist())
        grad_norm = np.linalg.norm(g)
        if grad_norm < tol: print(f"Converged at iteration {i}: grad_norm={grad_norm:.3e}"); break

        #D = np.diag(np.maximum(np.abs(np.diag(H)), 1e-8))  # Prevent zero diagonals
        try: delta = -g #np.linalg.solve(H + lambda_current * D, -g)
        except np.linalg.LinAlgError: lambda_current *= 10; continue

        # Evaluate new point
        x_new = x + delta
        f_new = objective(x_new.tolist())[0]
        f_new, f_x = sum(f_new), sum(f_x) 

        #if f_new < f_x:
            #lambda_current = max(lambda_current / 10, min_lambda)
        x = x_new
            #print(f"Iter {i}: f={f_x:.6e} → {f_new:.6e}, "f"|g|={grad_norm:.3e}, λ={lambda_current:.1e}")
        #else: 
        #    lambda_current = min(lambda_current * 10, max_lambda)
        #    #print(f"Iter {i}: Rejected (f {f_new:.6e} > {f_x:.6e}), " f"λ={lambda_current:.1e}")
    return x

# Initial guesses (GeV)
#initial_masses = np.array([142.703747, 76.765395, 164.548815, 93.568769])
initial_masses = np.array([172.62, 80.385, 172.62, 80.385])
# Data in GeV

# Run optimization
optimized_masses = levenberg_marquardt(initial_masses, lambda0=1e-2, max_iter=100000, tol=1e-6, min_lambda=1e-2, max_lambda=1e10)
print("Optimized masses:", optimized_masses)


