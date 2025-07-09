import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

def compute_z2_coeffs(b, mu, m_T):
    px_b, py_b, pz_b, E_b = b
    px_mu, py_mu, pz_mu, E_mu = mu
    
    p_b = np.sqrt(px_b**2 + py_b**2 + pz_b**2)
    p_mu = np.sqrt(px_mu**2 + py_mu**2 + pz_mu**2)
    
    m_b2 = E_b**2 - p_b**2
    m_mu2 = E_mu**2 - p_mu**2
    
    beta_b = p_b / E_b
    beta_mu = p_mu / E_mu
    
    vec_b = np.array([px_b, py_b, pz_b])
    vec_mu = np.array([px_mu, py_mu, pz_mu])
    dot_product = np.dot(vec_b, vec_mu)
    cos_theta = dot_product / (p_b * p_mu)
    
    sin_theta = np.sqrt(1 - cos_theta**2)
    D1 = (-m_mu2 + m_b2 - m_T**2) / (2*E_b * sin_theta * beta_b)
    D2  = -(E_mu * beta_mu / (E_b * beta_b) + cos_theta) / sin_theta
    w = (beta_mu/beta_b - cos_theta) / sin_theta
    Om2 = w**2 + 1 - beta_mu**2
    
    P = 1 - (1 + w*D2)/Om2
    Q = -w*D1/Om2

    D2_minus_w = D2 - w
    A = P**2 * Om2 - D2_minus_w**2 + beta_mu**2
    B = 2*(-w*P*D1 - D1*D2_minus_w + beta_mu*E_mu)
    C = Q**2 * Om2 - D1**2 + (m_mu2)
    return A, B, C

def compute_z2_coeffs_and_derivatives(b, mu, m_W, m_T):
    px_b, py_b, pz_b, E_b = b
    px_mu, py_mu, pz_mu, E_mu = mu

    vec_b = np.array([px_b, py_b, pz_b])
    vec_mu = np.array([px_mu, py_mu, pz_mu])
    dot_product = np.dot(vec_b, vec_mu)

    p_b = np.sqrt(px_b**2 + py_b**2 + pz_b**2)
    p_mu = np.sqrt(px_mu**2 + py_mu**2 + pz_mu**2)
    beta_b, beta_mu = p_b / E_b, p_mu / E_mu
    cos_theta = dot_product / (p_b * p_mu)
    sin_theta = np.sqrt(1 - cos_theta**2)
    w   = (beta_mu/beta_b - cos_theta) / sin_theta
    Om2 = w**2 + 1 - beta_mu**2

    m_b2 = E_b**2 - p_b**2
    m_mu2 = E_mu**2 - p_mu**2
  
    Sx = compute_Sx(b, mu, m_W)
    A, B, C = compute_z2_coeffs(b, mu, m_T)
    Z2 = A*Sx**2 + B*Sx + C
    Z = np.sqrt(Z2) if Z2 > 0.0 else -np.sqrt(abs(Z2))
#    if Z == 0: Z = 1.0

    D1  = (- m_mu2 + m_b2 - m_T**2) / (2*E_b * sin_theta * beta_b)
    D2  = -(E_mu * beta_mu / (E_b * beta_b) + cos_theta) / sin_theta
    P = 1 - (1 + w*D2)/Om2
    D2_minus_w = D2 - w

    Q = -w*D1/Om2
    dD1_dmT = -m_T / (E_b * sin_theta * beta_b)
    dQ_dmT  = -w / Om2 * dD1_dmT
    
    dA_dmT = 0.0
    dB_dmT = 2*(P*Om2*dQ_dmT - D2_minus_w*dD1_dmT)
    dC_dmT = 2*(Q*Om2*dQ_dmT - D1*dD1_dmT)
    return A, B, C, dA_dmT, dB_dmT, dC_dmT, Sx, Z

def compute_Sx(b, mu, m_W):
    _, _, _, E_b = b
    px_mu, py_mu, pz_mu, E_mu = mu
    p_mu = np.sqrt(px_mu**2 + py_mu**2 + pz_mu**2)
    beta_mu = p_mu / E_mu
    m_mu2 = E_mu**2 - p_mu**2
    
    x0 = (m_mu2 - m_W**2) / (2 * E_mu)
    return (x0 * beta_mu - p_mu * (1 - beta_mu**2)) / beta_mu**2

def compute_mW2(Sx, b, mu):
    _, _, _, E_b = b
    px_mu, py_mu, pz_mu, E_mu = mu
    p_mu = np.sqrt(px_mu**2 + py_mu**2 + pz_mu**2)
    beta_mu = p_mu / E_mu
    m_mu2 = E_mu**2 - p_mu**2
    Sx = Sx * (beta_mu**2)
    Sx = Sx + p_mu * (1 - beta_mu**2)
    Sx = Sx / beta_mu
    Sx = 2*E_mu*Sx
    Sx = -(Sx - m_mu2)
    return Sx

def dZ_dmT(b, mu, m_W, m_T):
    A, B, C, dA_dmT, dB_dmT, dC_dmT, Sx, Z = compute_z2_coeffs_and_derivatives(b, mu, m_W, m_T)
    dZ2_dmT = dB_dmT * Sx + dC_dmT
    return dZ2_dmT / (2 * Z)

def dZ_dmW(b, mu, m_W, m_T):
    px_mu, py_mu, pz_mu, E_mu = mu
    p_mu = np.sqrt(px_mu**2 + py_mu**2 + pz_mu**2)
    dSx_dmW = -m_W / p_mu
    
    A, B, C = compute_z2_coeffs(b, mu, m_T)
    Sx = compute_Sx(b, mu, m_W)
    Z2 = A*Sx**2 + B*Sx + C
    Z = np.sqrt(Z2) if Z2 > 0 else -np.sqrt(abs(Z2)) 
#    if Z == 0: Z = 1.0

    dZ2_dmW = (2*A*Sx + B) * dSx_dmW
    return dZ2_dmW / (2 * Z) 

def compute_mW2_roots(b, mu, m_W, m_T):
    A, B, C = compute_z2_coeffs(b, mu, m_T)
    discriminant = B**2 - 4*A*C
    sqrt_disc = np.sqrt(discriminant)
    Sx1 = (-B + sqrt_disc) / (2*A)
    Sx2 = (-B - sqrt_disc) / (2*A)
    w1, w2 = compute_mW2(Sx1, b, mu), compute_mW2(Sx2, b, mu)
    return w1**0.5, w2**0.5

def compute_mT2_roots(b, mu, Sx):
    px_b, py_b, pz_b, E_b = b
    px_mu, py_mu, pz_mu, E_mu = mu
    
    p_b = np.sqrt(px_b**2 + py_b**2 + pz_b**2)
    p_mu = np.sqrt(px_mu**2 + py_mu**2 + pz_mu**2)
    m_b2 = E_b**2 - p_b**2
    m_mu2 = E_mu**2 - p_mu**2
    beta_b = p_b / E_b
    beta_mu = p_mu / E_mu
    
    vec_b = np.array([px_b, py_b, pz_b])
    vec_mu = np.array([px_mu, py_mu, pz_mu])
    dot_product = np.dot(vec_b, vec_mu)
    cos_theta = dot_product / (p_b * p_mu)
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    K1_base = (m_mu2 - 2*E_mu**2*(1 - beta_mu**2) + m_b2)
    K1_1 = -1/(2*E_b)
    D11 = K1_1 / (sin_theta * beta_b)
    D10 = K1_base / (2*E_b * sin_theta * beta_b)
    
    K2 = (-E_mu*beta_mu) / E_b
    D2 = (K2 / beta_b - cos_theta) / sin_theta
    w = (beta_mu/beta_b - cos_theta) / sin_theta
    Om2 = w**2 + 1 - beta_mu**2
    P = 1 - (1 + w*D2)/Om2
    
    Q1 = -w * D11 / Om2
    Q0 = -w * D10 / Om2
    D2w = D2 - w
    
    # Coefficients for u = m_T^2
    A_val = P**2 * Om2 - D2w**2 + beta_mu**2
    B0 = 2*(P*Q0*Om2 - D10*D2w + beta_mu*E_mu)
    B1 = 2*(P*Q1*Om2 - D11*D2w)
    C0 = Q0**2 * Om2 - D10**2 + (m_mu2)
    C1 = 2*(Q0*Q1*Om2 - D10*D11)
    C2 = Q1**2 * Om2 - D11**2
    
    a = C2
    b = C1 + B1 * Sx
    c = A_val * Sx**2 + B0 * Sx + C0
    discriminant = b**2 - 4*a*c
    
    sqrt_disc = discriminant**0.5
    u1 = (-b + sqrt_disc) / (2*a)
    u2 = (-b - sqrt_disc) / (2*a)
    return u1**0.5, u2**0.5



class NeutrinoSolution:
    def __init__(self, b, mu, m_W, m_T, t):
        self.b   = b
        self.t   = t  
        self.mu  = mu
        self.m_W = m_W
        self.m_T = m_T
        
        self._compute_kinematics()
        self._compute_Sx_Sy()
        self._compute_w_Om()
        self._compute_x1_y1()
        self._compute_Z()
        self._compute_rotation()
        self._compute_H_tilde()
        
    def _compute_kinematics(self):
        px_b ,  py_b,  pz_b, E_b  = self.b
        px_mu, py_mu, pz_mu, E_mu = self.mu
        
        # Momenta and masses
        self.p_b = np.sqrt(px_b**2 + py_b**2 + pz_b**2)
        self.p_mu = np.sqrt(px_mu**2 + py_mu**2 + pz_mu**2)
        self.m_b2 = E_b**2 - self.p_b**2
        self.m_mu2 = E_mu**2 - self.p_mu**2
        
        # Beta factors
        self.beta_b = self.p_b / E_b
        self.beta_mu = self.p_mu / E_mu
        
        # Angle between b and mu
        vec_b = np.array([px_b, py_b, pz_b])
        vec_mu = np.array([px_mu, py_mu, pz_mu])
        self.cos_theta = np.dot(vec_b, vec_mu) / (self.p_b * self.p_mu)
        self.sin_theta = np.sqrt(1 - self.cos_theta**2)
        
    def _compute_Sx_Sy(self):
        self.Sx = compute_Sx(self.b, self.mu, self.m_W)
        x0p = -(self.m_T**2 - self.m_W**2 - self.m_b2) / (2 * self.b[-1])
        self.Sy = (x0p / self.beta_b - self.cos_theta * self.Sx) / self.sin_theta
        
    def _compute_w_Om(self):
        self.w = (self.beta_mu/self.beta_b - self.cos_theta) / self.sin_theta
        self.Om2 = self.w**2 + 1 - self.beta_mu**2
        self.Om = np.sqrt(self.Om2)

    @property
    def ellipse_property(self):
        H = self.R_T.dot(self.H_tilde)
        A = H[:, 0]
        B = H[:, 1]
        
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
            'centroid': H[:, 2],
            'normal': N_normalized,
            'semi_major': major,
            'semi_minor': minor,
            'area': area
        }



    def _compute_x1_y1(self):
        Sx_w_Sy = self.Sx + self.w * self.Sy
        self.x1 = self.Sx - Sx_w_Sy / self.Om2
        self.y1 = self.Sy - Sx_w_Sy * self.w / self.Om2
        
    def _compute_Z(self):
        A, B, C = compute_z2_coeffs(self.b, self.mu, self.m_T)
        Z2 = A*self.Sx**2 + B*self.Sx + C
        self.Z = np.sqrt(Z2) if Z2 > 0 else np.sqrt(abs(Z2))
#        if self.Z == 0: self.Z = 1.0
        
    def _compute_rotation(self):
        def rotation_z(phi):
            c, s = np.cos(phi), np.sin(phi)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            
        def rotation_y(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            
        def rotation_x(psi):
            c, s = np.cos(psi), np.sin(psi)
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        
        # Muon angles in lab frame
        px_mu, py_mu, pz_mu, _ = self.mu
        phi_mu = np.arctan2(py_mu, px_mu)
        theta_mu = np.arctan2(np.sqrt(px_mu**2 + py_mu**2), pz_mu)
        
        # Rotation components
        R_z = rotation_z(-phi_mu)
        R_y = rotation_y(0.5*np.pi - theta_mu)
        
        # Rotate b-vector
        b_vec = np.array([self.b[i] for i in range(3)])
        b_rot = R_y @ (R_z @ b_vec)
        
        # X-rotation to align b-vector
        psi = -np.arctan2(b_rot[2], b_rot[1])
        R_x = rotation_x(psi)
        
        # Full rotation matrix (F-frame to lab)
        self.R_T = R_z.T @ R_y.T @ R_x.T
        
    def _compute_H_tilde(self):
        self.H_tilde = np.array([
            [self.Z/self.Om       , 0     , self.x1 - self.p_mu],
            [self.w*self.Z/self.Om, 0     ,             self.y1],
            [0                    , self.Z,                   0]
        ])

        print(self.R_T.dot(self.H_tilde))
        exit()

        
    def p_nu(self):    return self.R_T.dot(self.H_tilde).dot(np.array([np.cos(self.t), np.sin(self.t), 1]))
    def dSx_dmW(self): return -self.m_W / (self.mu[-1] * self.beta_mu)
    def dSy_dmW(self): return (self.m_W/(self.b[-1]*self.beta_b) - self.cos_theta*self.dSx_dmW()) / self.sin_theta
    def dSy_dmT(self): return (-self.m_T/(self.b[-1]*self.beta_b)) / self.sin_theta
    def dx1_dmW(self): return self.dSx_dmW()*(1 - 1/self.Om2) - (self.w/self.Om2)*self.dSy_dmW()
    def dy1_dmW(self): return self.dSy_dmW()*(1 - self.w**2/self.Om2) - (self.w/self.Om2)*self.dSx_dmW()
    def dx1_dmT(self): return -(self.w/self.Om2) * self.dSy_dmT()
    def dy1_dmT(self): return self.dSy_dmT() * (1 - self.w**2/self.Om2)
        
    def dH_tilde_dmW(self):
        dZ_dmW_ = dZ_dmW(self.b, self.mu, self.m_W, self.m_T)
        return np.array([
            [dZ_dmW_/self.Om       , 0      , self.dx1_dmW()],
            [self.w*dZ_dmW_/self.Om, 0      , self.dy1_dmW()],
            [0                     , dZ_dmW_,       0]
        ])
        
    def dH_tilde_dmT(self):
        dZ_dmT_ = dZ_dmT(self.b, self.mu, self.m_W, self.m_T)
        return np.array([
            [dZ_dmT_/self.Om       , 0      , self.dx1_dmT()],
            [self.w*dZ_dmT_/self.Om, 0      , self.dy1_dmT()],
            [0                     , dZ_dmT_,              0]
        ])
        
    def dp_nu_dmW(self): return self.R_T.dot(self.dH_tilde_dmW()).dot(np.array([np.cos(self.t), np.sin(self.t), 1]))
    def dp_nu_dmT(self): return self.R_T.dot(self.dH_tilde_dmT()).dot(np.array([np.cos(self.t), np.sin(self.t), 1]))
    def dp_nu_dt(self):  return self.R_T.dot(self.H_tilde).dot(np.array([-np.sin(self.t), np.cos(self.t), 0]))


class LevenbergMarquardt:
    """Levenberg-Marquardt optimization for neutrino distance minimization"""
    def __init__(self, b1, mu1, b2, mu2, params0, lambda0=1e-2, max_iter=10000, tol=1e-6):
        self.b1 = b1
        self.mu1 = mu1
        self.b2 = b2
        self.mu2 = mu2
        self.params = np.array(params0)
        self.lambda_val = lambda0
        self.max_iter = max_iter
        self.tol = tol
        self.cost_history = []
        
    def residual(self, params):
        """Compute residual vector: p_nu2 - p_nu1"""
        m_W1, m_T1, t1, m_W2, m_T2, t2 = params
        nu1 = NeutrinoSolution(self.b1, self.mu1, m_W1, m_T1, t1)
        nu2 = NeutrinoSolution(self.b2, self.mu2, m_W2, m_T2, t2)
        return nu2.p_nu() - nu1.p_nu()
    
    def jacobian(self, params):
        """Compute Jacobian of residual vector (3x6 matrix)"""
        m_W1, m_T1, t1, m_W2, m_T2, t2 = params
        nu1 = NeutrinoSolution(self.b1, self.mu1, m_W1, m_T1, t1)
        nu2 = NeutrinoSolution(self.b2, self.mu2, m_W2, m_T2, t2)
        
        J = np.zeros((3, 6))
        
        # Derivatives for first neutrino (negative sign)
        J[:, 0] = -2*nu1.dp_nu_dmW()  # dm_W1
        J[:, 1] = -2*nu1.dp_nu_dmT()  # dm_T1
        J[:, 2] = -2*nu1.dp_nu_dt()   # dt1
        
        # Derivatives for second neutrino (positive sign)
        J[:, 3] = 2*nu2.dp_nu_dmW()   # dm_W2
        J[:, 4] = 2*nu2.dp_nu_dmT()   # dm_T2
        J[:, 5] = 2*nu2.dp_nu_dt()    # dt2
        
        return J
    
    def cost(self, r):
        return 0.5 * np.sum(r**2)
    
    def step(self):
        """Perform single LM iteration"""
        r = self.residual(self.params)
        current_cost = self.cost(r)
        
        J = self.jacobian(self.params)
        JtJ = J.T.dot(J)
        Jtr = J.T.dot(r)
       
        diag = np.diag(np.diag(JtJ))
        delta = np.linalg.solve(JtJ + self.lambda_val * diag, -Jtr)
        #v = JtJ + self.lambda_val*diag
        #v = np.linalg.inv(v).dot(-Jtr)
        new_params = self.params + delta
       
        r_new = self.residual(new_params)
        new_cost = self.cost(r_new)
        
        # Gain ratio: actual vs predicted reduction
        predicted_reduction = -delta.dot(Jtr) - 0.5 * delta.dot(JtJ).dot(delta)
        actual_reduction = current_cost - new_cost
        rho = actual_reduction / (predicted_reduction + 1e-16) if predicted_reduction > 0 else 0
       
        # Update parameters and lambda
        if rho > 0:
            self.params = new_params
            self.lambda_val *= max(1/3, 1 - (2*rho - 1)**3)
            return new_cost, True
        else:
            self.lambda_val *= 2
            return current_cost, False
    
    def optimize(self):
        """Run full optimization"""
        for i in range(self.max_iter):
            cost_val, success = self.step()
            self.cost_history.append(cost_val)
            if cost_val < self.tol: break
        return self.params, np.array(self.cost_history)

def plot_optimization(cost_history):
    """Plot convergence of cost function"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(cost_history, 'b-o', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Cost (0.5 * ||r||²)')
    plt.title('Levenberg-Marquardt Convergence')
    plt.grid(True)
    plt.savefig('lm_convergence.png', dpi=150)
    plt.show()

def plot_neutrino_ellipses(b1, mu1, b2, mu2, params, n_points=100):
    """3D visualization of neutrino solution ellipses"""
    m_W1, m_T1, t1, m_W2, m_T2, t2 = params
    
    # Create solution objects
    nu1 = NeutrinoSolution(b1, mu1, m_W1, m_T1, 0)
    nu2 = NeutrinoSolution(b2, mu2, m_W2, m_T2, 0)
    
    # Generate ellipse points
    angles = np.linspace(0, 2*np.pi, n_points)
    points1 = np.array([nu1.R_T.dot(nu1.H_tilde).dot([np.cos(a), np.sin(a), 1]) for a in angles])
    points2 = np.array([nu2.R_T.dot(nu2.H_tilde).dot([np.cos(a), np.sin(a), 1]) for a in angles])
    
    # Optimized points
    opt_nu1 = NeutrinoSolution(b1, mu1, m_W1, m_T1, t1)
    opt_nu2 = NeutrinoSolution(b2, mu2, m_W2, m_T2, t2)
    p1_opt = opt_nu1.p_nu()
    p2_opt = opt_nu2.p_nu()
    
    # Create plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ellipses
    ax.plot(points1[:,0], points1[:,1], points1[:,2], 'b-', label='Neutrino 1 Solution Ellipse')
    ax.plot(points2[:,0], points2[:,1], points2[:,2], 'r-', label='Neutrino 2 Solution Ellipse')
    
    # Plot optimized points
    ax.scatter(*p1_opt, s=100, c='blue', marker='o', label=f'ν1: t={t1:.2f} rad')
    ax.scatter(*p2_opt, s=100, c='red', marker='o', label=f'ν2: t={t2:.2f} rad')
    
    # Plot distance vector
    dist = np.linalg.norm(p2_opt - p1_opt)
    ax.plot([p1_opt[0], p2_opt[0]], 
            [p1_opt[1], p2_opt[1]], 
            [p1_opt[2], p2_opt[2]], 
            'k--', linewidth=2, label=f'Distance: {dist:.2f} GeV')
    
    # Formatting
    ax.set_xlabel('Px [GeV]')
    ax.set_ylabel('Py [GeV]')
    ax.set_zlabel('Pz [GeV]')
    ax.set_title(f'Optimized Solution: m_W1={m_W1:.1f}, m_T1={m_T1:.1f}, m_W2={m_W2:.1f}, m_T2={m_T2:.1f}')
    ax.legend()
    plt.tight_layout()
    #plt.savefig('neutrino_ellipses_lm.png', dpi=150)
    return fig, ax


# Example usage
if __name__ == "__main__":
    b1  = np.array([-19.766428, -40.022249,   69.855886, 83.191328])
    mu1 = np.array([-14.306453, -47.019613,    3.816470, 49.295996])
    b2  = np.array([107.795878, -185.326183,  -67.989162, 225.794953])
    mu2 = np.array([4.941336  , -104.097506, -103.640669, 146.976547])
    params = np.array([80.385, 172.62, 0.0, 80.385, 172.62, 0.0]) 
    
    lm = LevenbergMarquardt(b1, mu1, b2, mu2, params)
    opt_params, cost_history = lm.optimize()
    
    m_W1, m_T1, t1, m_W2, m_T2, t2 = opt_params
    print(f"Optimized parameters:")
    print(f"  m_W1 = {m_W1:.4f} GeV, m_T1 = {m_T1:.4f} GeV, t1 = {t1:.4f} rad")
    print(f"  m_W2 = {m_W2:.4f} GeV, m_T2 = {m_T2:.4f} GeV, t2 = {t2:.4f} rad")
    print(f"Final cost: {cost_history[-1]:.6f}")
    
    plot_optimization(cost_history)
    plot_neutrino_ellipses(b1, mu1, b2, mu2, opt_params)
    plt.show()






#if __name__ == "__main__":
#    # Define input parameters (example values)
#    b1 = np.array([-19.766428, -40.022249,   69.855886, 83.191328])
#    mu1 = np.array([-14.306453, -47.019613,    3.816470, 49.295996])
#    b2 = np.array([107.795878, -185.326183,  -67.989162, 225.794953])
#    mu2 = np.array([4.941336  , -104.097506, -103.640669, 146.976547])
#    params = np.array([80.385, 172.62, 80.385, 172.62, 0.0, 0.0]) 
#    
#    # Create neutrino solutions
#    nu1 = NeutrinoSolution(b1, mu1, params[0], params[1], params[4])
#    nu2 = NeutrinoSolution(b2, mu2, params[2], params[3], params[5])
#    jac = jacobian_d2(nu1, nu2)
#    print("Jacobian:", jac)
#
#    A1, B1, C1 = compute_z2_coeffs(b1, mu1, params[1])
#    A2, B2, C2 = compute_z2_coeffs(b2, mu2, params[3])
#    Sx1 = compute_Sx(b1, mu1, params[0])
#    Sx2 = compute_Sx(b2, mu2, params[2])
#    Z2_1 = A1*Sx1**2 + B1*Sx1 + C1
#    Z2_2 = A2*Sx2**2 + B2*Sx2 + C2
#
#    dZ_dmT1 = dZ_dmT(b1, mu1, params[0], params[1])
#    dZ_dmT2 = dZ_dmT(b2, mu2, params[2], params[3])
#    dZ_dmW1 = dZ_dmW(b1, mu1, params[0], params[1]) 
#    dZ_dmW2 = dZ_dmW(b2, mu2, params[2], params[3]) 
#    r1_W1, r2_W1 = compute_mW2_roots(b1, mu1, params[0], params[1])
#    r1_W2, r2_W2 = compute_mW2_roots(b2, mu2, params[0], params[1])
#    r1_T1, r2_T1 = compute_mT2_roots(b1, mu1, Sx1)
#    r1_T2, r2_T2 = compute_mT2_roots(b2, mu2, Sx2)
#
#    print(f"Z^2: {Z2_1:.6f}, {Z2_2:.6f}")
#    print(f"dZ/dmT: {dZ_dmT1:.6f}, {dZ_dmT2:.6f}")
#    print(f"dZ/dmW: {dZ_dmW1:.6f}, {dZ_dmW2:.6f}")
#    print(f"root W: {r1_W1:.6f}, {r2_W1:.6f}, {r1_W2:.6f}, {r2_W2:.6f}")
#    print(f"root T: {r1_T1:.6f}, {r2_T1:.6f}, {r1_T2:.6f}, {r2_T2:.6f}")
