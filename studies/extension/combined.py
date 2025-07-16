import numpy as np
import math

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



def get_mt2(sl, timeout = None):
    eb, em, bb, bm   = sl.b[-1], sl.mu[-1], sl.beta_b, sl.beta_mu
    mb, mm, mt1, mw1 = sl.m_b2**0.5, sl.m_mu2**0.5, sl.m_T, sl.m_W
    sin_th = sl.s
    cos_th = sl.c

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



    w = (bm / bb - cos_th) / sin_th
    omega = w**2 + 1 - bm**2

    a_sy = -1 / (2 * eb * bb * sin_th)
    b_sy = ((mw1**2 + mb) / (2 * eb * bb) - cos_th * sx_) / sin_th

    a_x1 = - (w * a_sy) / omega
    b_x1 = ((omega - 1) * sx_ - w * b_sy) / omega
    a    = omega * a_x1**2 - a_sy**2
    b    = 2 * omega * a_x1 * b_x1 - 2 * a_sy * (b_sy - w * sx_)
    c    = omega * b_x1**2 - (b_sy - w * sx_)**2 - cons- sl.Z2
    discriminant = b**2 - 4 * a * c

    root1 = (-b + math.sqrt(discriminant)) / (2 * a)
    root2 = (-b - math.sqrt(discriminant)) / (2 * a)
   
    root1 = math.sqrt(root1) if root1 >= 0 else mt1
    root2 = math.sqrt(root2) if root2 >= 0 else mt1
    mot = min(root1, root2)


def get_mW(sl): #):
    sin_theta = sl.sin_theta #math.sin(theta)
    cos_theta = sl.cos_theta #math.cos(theta)
    Eb, Em = sl.b[-1], sl.mu[-1]
    Bb, Bm, mb, mm, mT = sl.beta_b, sl.beta_mu, sl.m_b2**0.5, sl.m_mu2**0.5, sl.m_T

    w = (Bm / Bb - cos_theta) / sin_theta
    om2 = w**2 + 1 - Bm**2
    
    E0 = mm**2 / (2 * Em)
    E1 = -1 / (2 * Em)
    
    P1 = 1 / (2 * Eb)
    P0 = (mb**2 - mT**2) / (2 * Eb)
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

    def get_mW(self): return get_mW(self)

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
        self.Z2 = Z2
        self.Z = np.sqrt(Z2) if Z2 > 0 else np.sqrt(abs(Z2))

    def dSx_dmW(self): return -self.m_W / (self.mu[-1] * self.beta_mu)
    def dSy_dmW(self): return (self.m_W/(self.b[-1]*self.beta_b) - self.cos_theta*self.dSx_dmW()) / self.sin_theta
    def dSy_dmT(self): return (-self.m_T/(self.b[-1]*self.beta_b)) / self.sin_theta
    def dx1_dmW(self): return self.dSx_dmW()*(1 - 1/self.Om2) - (self.w/self.Om2)*self.dSy_dmW()
    def dy1_dmW(self): return self.dSy_dmW()*(1 - self.w**2/self.Om2) - (self.w/self.Om2)*self.dSx_dmW()
    def dx1_dmT(self): return -(self.w/self.Om2) * self.dSy_dmT()
    def dy1_dmT(self): return self.dSy_dmT() * (1 - self.w**2/self.Om2)
       
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
       
    def H(self): return self.R_T.dot(self.H_tilde)
    def p_nu(self):      return self.R_T.dot(self.H_tilde).dot(np.array([np.cos(self.t), np.sin(self.t), 1]))
    def dp_nu_dt(self):  return self.R_T.dot(self.H_tilde).dot(np.array([-np.sin(self.t), np.cos(self.t), 0]))
    def dp_nu_dmW(self): return self.R_T.dot(self.dH_tilde_dmW()).dot(np.array([np.cos(self.t), np.sin(self.t), 1]))
    def dp_nu_dmT(self): return self.R_T.dot(self.dH_tilde_dmT()).dot(np.array([np.cos(self.t), np.sin(self.t), 1]))
    def H_perp(self):    return np.vstack([self.H()[:2], [0, 0, 1]])
    def K(self):         return self.H().dot(np.linalg.inv(self.H_perp()))



    def misc(self):
        print("--------- Neutrino ------------");
        #print("Sx:      ", compute_Sx(self.b, self.mu, self.m_W))
        #print("dSx_dmW: ", self.dSx_dmW())
        #print("Sy:      ", self.Sy)         
        #print("dSy_dmW: ", self.dSy_dmW())   
        #print("dSy_dmT: ", self.dSy_dmT())   
        #print("w:       ", self.w        )   
        #print("w2:      ", self.w**2     )   
        #print("om2:     ", self.Om2      )   
        print("Z:       ", self.Z      )   
        #print("Z2:      ", self.Z2     )   
        #print("dZ_dmT:  ", dZ_dmT(self.b, self.mu, self.m_W, self.m_T))   
        #print("dZ_dmW:  ", dZ_dmW(self.b, self.mu, self.m_W, self.m_T))   
        #print("x1:      ", self.x1     )   
        #print("dx1_dmW: ", self.dx1_dmW())   
        #print("dx1_dmT: ", self.dx1_dmT())   
        #print("y1:      ", self.y1     )   
        #print("dy1_dmW: ", self.dy1_dmW())   
        #print("dy1_dmT: ", self.dy1_dmT())   
        #print("mW root: ", compute_mW2_roots(self.b, self.mu, self.m_W, self.m_T))
        #print("mW derivative = 0: ", self.get_mW())
        #print("H                : \n", self.R_T.dot(self.H_tilde))
        print("Om:", self.Om)
        print("RT               : \n", self.R_T)
        print("H_Tilde          : \n", self.H_tilde)
        print("H_perp           : \n", self.H_perp)
        print("K                : \n", self.K())
        print("dH_dmW           : \n", self.R_T.dot(self.dH_tilde_dmW()))
        print("dH_dmT           : \n", self.R_T.dot(self.dH_tilde_dmT()))

        exit()



        #double A, B, C; 
        #this -> Z2_coeff(&A, &B, &C); 
        #std::cout << "Z2 A: " << A << " B: " << B << " C: " << C << std::endl; 

        #std::cout << "N" << std::endl;
        #print(this -> N()); 

        #std::cout << "H" << std::endl; 
        #print(this -> H()); 

        #std::cout << "H_perp" << std::endl; 
        #print(this -> H_perp());    

        #std::cout << "H_tilde" << std::endl; 
        #print(this -> H_tilde());

        #std::cout << "dH_dmW" << std::endl; 
        #print(this -> dH_dmW());

        #std::cout << "dH_dmT" << std::endl; 
        #print(this -> dH_dmT()); 

        #std::cout << "K" << std::endl; 
        #print(this -> K()); 

        #std::cout << "RT" << std::endl; 
        #print(this -> R_T());













if __name__ == "__main__":
    # Define input parameters (example values)
    b1 = np.array([-19.766428, -40.022249,   69.855886, 83.191328])
    b2 = np.array([107.795878, -185.326183,  -67.989162, 225.794953])
    mu1 = np.array([-14.306453, -47.019613,    3.816470, 49.295996])
    mu2 = np.array([4.941336  , -104.097506, -103.640669, 146.976547])
    params = np.array([80.385, 172.62, 80.385, 172.62, 0.0, 0.0]) 
    
    # Create neutrino solutions
    nu1 = NeutrinoSolution(b1, mu1, params[0], params[1], params[4])
    nu2 = NeutrinoSolution(b2, mu2, params[2], params[3], params[5])

    nu1.misc()
    nu2.misc()




















