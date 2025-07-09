import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class LorentzVector:
    def __init__(self, px, py, pz, E):
        self.px = px
        self.py = py
        self.pz = pz
        self.E = E
        self.vec = np.array([px, py, pz])
    
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
    
    def dot(self, other):
        return np.dot(self.vec, other.vec)
    
    def angle(self, other):
        if self.p == 0 or other.p == 0:
            return 0
        cos_theta = self.dot(other) / (self.p * other.p)
        return math.acos(np.clip(cos_theta, -1, 1))
    
    def __repr__(self):
        return f"LorentzVector(px={self.px:.2f}, py={self.py:.2f}, pz={self.pz:.2f}, E={self.E:.2f})"

class NuEllipseCalculator:
    """Computes neutrino ellipse parameters for a single top decay"""
    def __init__(self, b, mu, m_b, m_mu):
        self.b = b
        self.mu = mu
        self.m_b = m_b
        self.m_mu = m_mu
        
        # Compute angles between b and mu
        angle_b_mu = b.angle(mu)
        self.costh = math.cos(angle_b_mu)
        self.sinth = math.sin(angle_b_mu)
        
        # Compute muon angles
        self.phi_mu = math.atan2(mu.py, mu.px)
        self.theta_mu = math.acos(mu.pz / mu.p) if mu.p > 0 else 0
    
    def compute_R_T(self):
        """Compute rotation matrix from F frame to lab frame (Section 2.5)"""
        # Step 1: Rotate around z-axis by φ_μ
        R_z = np.array([
            [math.cos(self.phi_mu), -math.sin(self.phi_mu), 0],
            [math.sin(self.phi_mu), math.cos(self.phi_mu), 0],
            [0, 0, 1]
        ])
        
        # Step 2: Rotate around y'-axis by (θ_μ - π/2)
        angle_y = self.theta_mu - math.pi/2
        R_y = np.array([
            [math.cos(angle_y), 0, math.sin(angle_y)],
            [0, 1, 0],
            [-math.sin(angle_y), 0, math.cos(angle_y)]
        ])
        
        # Step 3: Rotate around x''-axis to align b in xy-plane
        b_double_prime = (R_y @ R_z) @ self.b.vec
        if np.linalg.norm(b_double_prime[1:3]) < 1e-8:
            alpha = 0
        else:
            alpha = math.atan2(b_double_prime[2], b_double_prime[1])
        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(alpha), -math.sin(alpha)],
            [0, math.sin(alpha), math.cos(alpha)]
        ])
        
        # Combined rotation: R = R_z(φ_μ) * R_y'(θ_μ - π/2) * R_x''(α)
        R_T = R_x @ R_y @ R_z
        return R_T
    
    def compute_ellipse(self, mT, mW):
        """Compute ellipse parameters for given masses"""
        # 1. Compute key kinematic parameters
        Eb, Emu = self.b.E, self.mu.E
        beta_b = self.b.beta
        beta_mu = self.mu.beta
        gamma_mu_inv = self.m_mu / Emu
        
        # 2. Compute intermediate terms
        x0 = -0.5/Emu * (mW**2 - self.m_mu**2)
        x0p = -0.5/Eb * (mT**2 - mW**2 - self.m_b**2)
        
        # 3. Compute Sx and Sy
        Sx = (x0 * beta_mu - self.mu.p * gamma_mu_inv**2) / beta_mu**2
        Sy = (x0p / beta_b - self.costh * Sx) / self.sinth
        
        # 4. Compute omega and Omega
        omega = (beta_mu / beta_b - self.costh) / self.sinth
        Omega2 = omega**2 + gamma_mu_inv**2
        
        # 5. Compute x1, y1
        x1 = Sx - (Sx + omega * Sy) / Omega2
        y1 = Sy - (Sx + omega * Sy) * omega / Omega2
        
        # 6. Handle degenerate cases
        Z_val = 0
        if Omega2 > 0:
            # 7. Compute Z² (but don't use as constraint)
            eps2 = gamma_mu_inv**2 * (mW**2 - 0)  # m_nu=0
            Z2 = x1**2 * Omega2 - (Sy - omega * Sx)**2 - (mW**2 - x0**2 - eps2)
            Z_val = math.sqrt(max(0, Z2))
        
        Omega_val = math.sqrt(Omega2) if Omega2 > 0 else 0
        
        # 8. Compute H_tilde in F frame
        H_tilde = np.array([
            [Z_val/Omega_val if Omega_val > 0 else 0, 0, x1 - self.mu.px],
            [omega*Z_val/Omega_val if Omega_val > 0 else 0, 0, y1],
            [0, Z_val, 0]
        ])
        
        # 9. Compute rotation matrix to lab frame
        R_T = self.compute_R_T()
        
        # 10. Compute H in lab frame: H = R_T * H_tilde
        H_lab = R_T @ H_tilde
        
        # 11. Compute centroid (last column of H)
        centroid = H_lab[:, 2]
        
        return H_lab, centroid[:2], Z2

class NeutrinoEllipseOptimizer:
    """Optimizes masses to bring centroids close to MET position"""
    def __init__(self, b1, mu1, b2, mu2, met, lambda_reg=0.0):
        # Convert inputs to LorentzVector format
        self.b1 = LorentzVector(**b1)
        self.mu1 = LorentzVector(**mu1)
        self.b2 = LorentzVector(**b2)
        self.mu2 = LorentzVector(**mu2)
        self.met = np.array(met)
        self.lambda_reg = lambda_reg
        
        # Create ellipse calculators
        self.calc1 = NuEllipseCalculator(self.b1, self.mu1, self.b1.mass, self.mu1.mass)
        self.calc2 = NuEllipseCalculator(self.b2, self.mu2, self.b2.mass, self.mu2.mass)
   

    def compute_ellipses(self, masses):
        """Compute both ellipses for given mass parameters"""
        mT1, mW1, mT2, mW2 = masses
        H1, c1, Z2_1 = self.calc1.compute_ellipse(mT1, mW1)
        H2, c2, Z2_2 = self.calc2.compute_ellipse(mT2, mW2)
        return H1, H2, c1, c2, Z2_1, Z2_2
  
    def find_closest_points(self, H1, H2):
        """Find points on ellipses that minimize distance to MET constraint"""
        # Parametrize ellipses with angles
        theta = np.linspace(0, 2*np.pi, 100)
        t_vectors = np.array([np.cos(theta), np.sin(theta), np.ones_like(theta)])
        
        # Compute ellipse points
        ellipse1 = H1[:2, :] @ t_vectors
        ellipse2 = H2[:2, :] @ t_vectors
        
        # Find combination closest to MET
        min_dist = float('inf')
        best_p1 = None
        best_p2 = None
        
        for i in range(len(theta)):
            p1 = ellipse1[:, i]
            for j in range(len(theta)):
                p2 = ellipse2[:, j]
                dist  = sum(((p1 + p2) - 0*self.met)**2)
                #dist = np.linalg.norm(residual)
                if dist < min_dist:
                    min_dist = dist
                    best_p1 = p1
                    best_p2 = p2
        return best_p1, best_p2, min_dist

    def objective(self, masses):
        """Objective function: distance between midpoint and MET"""
        H1, H2, c1, c2, Z2_1, Z2_2 = self.compute_ellipses(masses)
        
        # Calculate midpoint between centroids
        midpoint = (c1 + c2) / 2
        p1, p2, dist = self.find_closest_points(H1, H2)
        #dist = sum(((p1 + p2)/2 - midpoint)**2)
        dist = sum(((p1 + p2)/2 - self.met)**2)
#        print((p1 + p2)/2, midpoint)

        # Distance from midpoint to MET
        #dist += np.sum((midpoint - self.met)**2)

        #reg_term = self.lambda_reg * (np.sum((masses - np.array([172.5, 80.4, 172.5, 80.4]))**2))
        reg_term = 0 #abs((math.log10(abs(min(1, Z2_2)))) + abs(math.log10(abs(min(1, Z2_1)))))
        print(masses, dist + reg_term, Z2_1, Z2_2)
        return dist + reg_term

    def gradient(self, masses):
        """Numerical gradient of objective function"""
        eps = 0.01  # Finite difference step
        grad = np.zeros(4)
        
        # Compute gradient for each parameter
        for i in range(4):
            masses_plus = masses.copy()
            masses_plus[i] += eps
            f_plus = self.objective(masses_plus)
            
            masses_minus = masses.copy()
            masses_minus[i] -= eps
            f_minus = self.objective(masses_minus)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad
    
    def optimize(self, initial_masses, bounds=None):
        """Optimize masses using L-BFGS-B algorithm"""
        
        for i in range(1000):
            res = minimize(
                fun=self.objective,
                x0=initial_masses,
                method='L-BFGS-B',
                jac=self.gradient,
#                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, "disp" : True}
            )
            initial_masses = res.x
            print(initial_masses)
            if i % 10 == 9: self.plot_ellipses(res.x, "Optimized Neutrino Ellipses")
        return res
   
    def plot_ellipses(self, masses, title="Neutrino Momentum Ellipses"):
        """Plot both neutrino ellipses in transverse plane"""
        H1, H2, c1, c2, Z2_1, Z2_2 = self.compute_ellipses(masses)
        
        # Find best points
        p1, p2, dist = self.find_closest_points(H1, H2)

        # Generate ellipse points
        theta = np.linspace(0, 2*np.pi, 100)
        t_vectors = np.array([np.cos(theta), np.sin(theta), np.ones_like(theta)])
        
        # Compute ellipse points for decay 1
        ellipse1 = H1[:2, :] @ t_vectors
        # Compute ellipse points for decay 2
        ellipse2 = H2[:2, :] @ t_vectors
        
        # Calculate midpoint
        midpoint = (c1 + c2) / 2
        
        plt.figure(figsize=(10, 8))
        plt.title(title)
        
        # Plot solution points
        plt.scatter(p1[0], p1[1], c='blue', s=150, marker='o', label='ν1 Solution')
        plt.scatter(p2[0], p2[1], c='red' , s=150, marker='o', label='ν2 Solution')

        # Plot ellipses
        plt.plot(ellipse1[0], ellipse1[1], 'b-', label='Neutrino 1', alpha=0.7)
        plt.plot(ellipse2[0], ellipse2[1], 'r-', label='Neutrino 2', alpha=0.7)
        plt.scatter(p1[0] + p2[0], p1[1] + p2[1], c='purple', marker='s', s=100, label='ν1 + ν2')

        # Plot centroids
        plt.scatter(c1[0], c1[1], c='blue', s=100, label='Centroid 1')
        plt.scatter(c2[0], c2[1], c='red' , s=100, label='Centroid 2')
        
        # Plot midpoint
        plt.scatter(midpoint[0], midpoint[1], c='purple', s=150, marker='+', label='Midpoint')
        
        # Plot MET position
        plt.scatter(self.met[0], self.met[1], c='green', marker='*', s=200, label='MET')
        
        # Draw vectors
        plt.arrow(0, 0, p1[0], p1[1], color='blue', width=0.5, alpha=0.5)
        plt.arrow(p1[0], p1[1], p2[0], p2[1], color='red', width=0.5, alpha=0.5)
        plt.arrow(0, 0, p1[0] + p2[0], p1[1] + p2[1], color='purple', width=0.5, alpha=0.7)
        
        # Draw line from vector sum to MET
        plt.plot([p1[0] + p2[0], self.met[0]], [p1[1] + p2[1], self.met[1]], 'k--', linewidth=1.5, label='Residual')

        # Draw lines connecting centroids and to MET
        plt.plot([c1[0], c2[0]], [c1[1], c2[1]], 'k-', alpha=0.5, label='Centroid Connection')
        plt.plot([midpoint[0], self.met[0]], [midpoint[1], self.met[1]], 'm--', linewidth=2, label='Midpoint-MET')
        
        plt.xlabel(r'$p_x$ (GeV)')
        plt.ylabel(r'$p_y$ (GeV)')
        plt.axhline(0, color='gray', linestyle='--', alpha=0.3)
        plt.axvline(0, color='gray', linestyle='--', alpha=0.3)
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
       

        # Add information box
        residual = self.met - (p1 + p2)
        residual_norm = np.linalg.norm(residual)

        # Add information box
        plt.figtext(0.15, 0.02, 
                   f"Midpoint-MET distance: {np.linalg.norm(midpoint - self.met):.2f} GeV\n"
                   f"Z² values: {Z2_1:.2f}, {Z2_2:.2f}\n"
                   f"Residual: {residual_norm:.2f} GeV\n"
                   f"Vector sum: ({p1[0]+p2[0]:.1f}, {p1[1]+p2[1]:.1f})\n",
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.show()


# Example usage with visualization
if __name__ == "__main__":
    # Define particle 4-vectors (px, py, pz, E)
    b1  = [-19.766428, -40.022249, 69.855886, 83.191328]
    b2  = [107.795878, -185.326183, -67.989162, 225.794953]
    mu1 = [-14.306453, -47.019613, 3.816470, 49.295996]
    mu2 = [4.941336, -104.097506, -103.640669, 146.976547]
    dc = ["px", "py", "pz", "E"]

    b1  = {dc[i]:  b1[i] for i in range(len(dc))}
    mu1 = {dc[i]: mu1[i] for i in range(len(dc))}
    b2  = {dc[i]:  b2[i] for i in range(len(dc))}
    mu2 = {dc[i]: mu2[i] for i in range(len(dc))}

    # MET measurement (missing transverse energy)
    met = [106.435841, -141.293331]
    
    # Create optimizer
    optimizer = NeutrinoEllipseOptimizer(
        b1=b1, mu1=mu1,
        b2=b2, mu2=mu2,
        met=met,
        lambda_reg=0.1  # Small regularization to keep masses physical
    )
    
    # Set bounds for masses (GeV)
    bounds = [
        (120, 250),   # mT1
        (50, 120),     # mW1
        (120, 250),   # mT2
        (50, 120)      # mW2
    ]
    
    # Initial mass guesses
    initial_masses = [172.5, 80.4, 172.5, 80.4]
    
    # Plot initial ellipses
    print("Plotting initial ellipses with masses:", initial_masses)
    optimizer.plot_ellipses(initial_masses, "Initial Neutrino Ellipses")
    
    # Run optimization
    print("\nStarting optimization...")
    result = optimizer.optimize(initial_masses, bounds=bounds)
    
    print("\nOptimization results:")
    print(f"Status: {result.message}")
    print(f"Optimal masses: mT1={result.x[0]:.3f}, mW1={result.x[1]:.3f}, "
          f"mT2={result.x[2]:.3f}, mW2={result.x[3]:.3f}")
    print(f"Final objective value: {result.fun:.6e}")
    
    # Plot final ellipses
    print("\nPlotting optimized ellipses...")
    optimizer.plot_ellipses(result.x, "Optimized Neutrino Ellipses")
