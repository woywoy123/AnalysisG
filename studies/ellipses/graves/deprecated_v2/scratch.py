import sympy as sp
from proofs.tests import *

mu, b = particle("mu"), particle("b")
theta = symbol("theta")
alpha, beta, gamma = symbols(["alpha", "beta", "gamma"])

wp_ = 1 / sp.sin(theta) * ( mu.beta/b.beta - sp.cos(theta))
wm_ = 1 / sp.sin(theta) * (-mu.beta/b.beta - sp.cos(theta))
op_ = sp.sqrt(wp_**2 + 1 - mu.beta ** 2)
om_ = sp.sqrt(wm_**2 + 1 - mu.beta ** 2)

sx = sp.cos(alpha) * sp.cos(beta) + sp.sin(alpha) * sp.sin(beta) * sp.cos(gamma)

GP = (wp_ + wm_) / op_**2
GM = (wp_ - wm_) / om_**2

r1 = (( op_ - om_) ** 2 - (wp_ + wm_) ** 2) / (2 * (wp_ + wm_) )
r2 = (( op_ + om_) ** 2 - (wp_ + wm_) ** 2) / (2 * (wp_ + wm_) )

l1 = (- GP * GM * (1 + r1 *r2) + GM * GP * ((r1 ** 2 + 1 )* ( r2 ** 2 +1 ))**0.5)
l2 = (- GP * GM * (1 + r1 *r2) - GM * GP * ((r1 ** 2 + 1 )* ( r2 ** 2 +1 ))**0.5)
x = - ( sp.sqrt(1 - sx**2) / ( (1 - mu.beta) * (1 + mu.beta)) * sx )

#x = sp.simplify(sp.cancel(l1 - l2).expand()).subs(sp.sin(theta) ** 2, 1 - sx ** 2).subs(sp.cos(theta), sx).expand()
x = sp.simplify(x).together().together().subs(sp.sin(theta), (1 - sx ** 2)**0.5).together()
sols = sp.solve(x, sp.cos(gamma))
sp.pprint(sols)














#wp_ = 1 / sp.sin(theta) * ( mu.beta/b.beta - sp.cos(theta))
#wm_ = 1 / sp.sin(theta) * (-mu.beta/b.beta - sp.cos(theta))
#op_ = sp.sqrt(wp_**2 + 1 - mu.beta ** 2)
#om_ = sp.sqrt(wm_**2 + 1 - mu.beta ** 2)
#
#lambda1 = (( op_ - om_) ** 2 - (wp_ + wm_) ** 2) / (2 * (wp_ + wm_) )
#lambda2 = (( op_ + om_) ** 2 - (wp_ + wm_) ** 2) / (2 * (wp_ + wm_) )
#
#x = sp.cancel(lambda1 - lambda2).subs(sp.sin(theta) ** 2, 1 - sx ** 2).expand(force = True)
#x = x.subs(sp.cos(theta), sx).expand(force = True).together().together().subs(sp.sin(theta), (1 - sx ** 2)**0.5).expand(force = True)
#sols = sp.solve(x, sp.cos(gamma))
#
#for i in sols:
#
#    print(i)
#    continue
#
#    for j in [lambda1, lambda2]:
#        k = j.subs(sp.cos(theta), sx).subs(sp.sin(theta), sp.sqrt(1 - sx**2)).together()#.expand(force = True)
#        k = k.subs(sp.cos(gamma), i).subs(mu.beta, beta_mass_energy(mu)).subs(b.beta, beta_mass_energy(b)).expand(force = True).together()
#        sp.pprint(k)
#
#        exit()
#
#    sp.pprint(i)












exit()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math

from atomics import *
from particle import *

def boost_to_rest_frame(particle, boost_particle):
    p_mu = np.array([particle.px, particle.py, particle.pz, particle.e])
    p_boost = np.array([boost_particle.px, boost_particle.py, boost_particle.pz, boost_particle.e])
    
    beta = np.array([boost_particle.px, boost_particle.py, boost_particle.pz]) / boost_particle.e
    beta2 = np.dot(beta, beta)
    gamma = 1.0 / math.sqrt(1.0 - beta2) if beta2 < 1 else 1.0
    
    b4 = np.outer(gamma * beta, gamma * beta) / (1.0 + gamma) if gamma > 1 else np.zeros((3,3))
    boost_matrix = np.eye(4)
    boost_matrix[:3, :3] = np.eye(3) + b4
    boost_matrix[:3, 3] = -gamma * beta
    boost_matrix[3, :3] = -gamma * beta
    
    # Apply boost
    p_rest = np.dot(boost_matrix, p_mu)
    return Particle(p_rest[0], p_rest[1], p_rest[2], p_rest[3])

def compute_ellipse_points(H, num_points=100):
    """Compute points on the ellipse defined by H matrix"""
    t_values = np.linspace(0, 2*np.pi, num_points)
    points = []
    
    for t in t_values:
        t_vec = np.array([np.cos(t), np.sin(t), 1])
        p_nu = H.dot(t_vec)
        points.append(p_nu)
    return np.array(points)

def plot_ellipse_projections(points, ax1, ax2, ax3, color='blue', label='Ellipse', linewidth=1.5):
    ax1.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth, label=label)
    ax1.set_xlabel('x (GeV)')
    ax1.set_ylabel('y (GeV)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # XZ projection
    ax2.plot(points[:, 0], points[:, 2], color=color, linewidth=linewidth)
    ax2.set_xlabel('x (GeV)')
    ax2.set_ylabel('z (GeV)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # YZ projection
    ax3.plot(points[:, 1], points[:, 2], color=color, linewidth=linewidth)
    ax3.set_xlabel('y (GeV)')
    ax3.set_ylabel('z (GeV)')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

def compute_decay_angles(top, b, lep, nu, W):
    angles = {}
    
    # 1. theta_tW: Angle between top and W in top rest frame
    top_rest = boost_to_rest_frame(W, top)
    angles['theta_tW'] = math.acos(costheta(top, top_rest))
    
    # 2. theta_bW: Angle between b and W in W rest frame
    b_rest = boost_to_rest_frame(b, W)
    angles['theta_bW'] = math.acos(costheta(W, b_rest))
    
    # 3. theta_lepnu: Angle between lepton and neutrino in W rest frame
    lep_rest = boost_to_rest_frame(lep, W)
    nu_rest = boost_to_rest_frame(nu, W)
    angles['theta_lepnu'] = math.acos(costheta(lep_rest, nu_rest))
    
    # 4. theta_bW_lab: Angle between b and W in lab frame
    angles['theta_bW_lab'] = math.acos(costheta(b, W))
    
    # 5. theta_lepnu_lab: Angle between lepton and neutrino in lab frame
    angles['theta_lepnu_lab'] = math.acos(costheta(lep, nu))

    # 4. theta_bW_lab: Angle between b and W in lab frame
    angles['theta_nuW_lab'] = math.acos(costheta(nu, W))

    # 4. theta_bW_lab: Angle between b and W in lab frame
    angles['theta_tW_lab'] = math.acos(costheta(top, W))

    return angles

def plot_truth_ellipses(nu_sol, top, b, lep, nu, W, output_prefix='ellipse_plots'):
    H_tilde = nu_sol.H_tilde  # In F frame
    H = nu_sol.H              # In lab frame
    
    points_tilde = compute_ellipse_points(H_tilde)
    points_lab = compute_ellipse_points(H)
    
    # Boost ellipse points to W rest frame
    points_W_rest = []
    for point in points_lab:
        temp_particle = Particle(point[0], point[1], point[2], math.sqrt(abs(point[0]**2 + point[1]**2 + point[2]**2 + nu.mass**2)))
        p_rest = boost_to_rest_frame(temp_particle, W)
        points_W_rest.append([p_rest.px, p_rest.py, p_rest.pz])
    points_W_rest = np.array(points_W_rest)
    
    # Compute decay angles
    angles = compute_decay_angles(top, b, lep, nu, W)
    
    # Create figure for lab frame
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle('Neutrino Ellipse in Lab Frame', fontsize=14, fontweight='bold')
    plot_ellipse_projections(points_lab, axes1[0], axes1[1], axes1[2], color='blue', label='Lab Frame Ellipse')
    axes1[0].plot(nu.px, nu.py, 'r*', markersize=10, label='Truth Neutrino')
    axes1[1].plot(nu.px, nu.pz, 'r*', markersize=10)
    axes1[2].plot(nu.py, nu.pz, 'r*', markersize=10)
    
    axes1[0].legend(loc='best')
    
    angle_text = f'θ_bW (lab): {angles["theta_bW_lab"]*180/np.pi:.1f}°\n'
    angle_text += f'θ_lepnu (lab): {angles["theta_lepnu_lab"]*180/np.pi:.1f}°'
    axes1[0].text(0.05, 0.95, angle_text, transform=axes1[0].transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    #plt.savefig(f'{output_prefix}_lab_frame.png', dpi=150, bbox_inches='tight')
    
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('Neutrino Ellipse in W Rest Frame', fontsize=14, fontweight='bold')
    
    plot_ellipse_projections(points_W_rest, axes2[0], axes2[1], axes2[2], 
                            color='green', label='W Rest Frame Ellipse')
    
    nu_W_rest = boost_to_rest_frame(nu, W)
    axes2[0].plot(nu_W_rest.px, nu_W_rest.py, 'r*', markersize=10, label='Truth Neutrino')
    axes2[1].plot(nu_W_rest.px, nu_W_rest.pz, 'r*', markersize=10)
    axes2[2].plot(nu_W_rest.py, nu_W_rest.pz, 'r*', markersize=10)
    axes2[0].legend(loc='best')
    
    angle_text = f'θ_tW: {angles["theta_tW"]*180/np.pi:.1f}°\n'
    angle_text += f'θ_bW: {angles["theta_bW"]*180/np.pi:.1f}°\n'
    angle_text += f'θ_lepnu: {angles["theta_lepnu"]*180/np.pi:.1f}°'
    axes2[0].text(0.05, 0.95, angle_text, transform=axes2[0].transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    #plt.tight_layout()
    #plt.savefig(f'{output_prefix}_W_rest_frame.png', dpi=150, bbox_inches='tight')
    
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle('Neutrino Ellipse in F Frame (H_tilde)', fontsize=14, fontweight='bold')
    
    plot_ellipse_projections(points_tilde, axes3[0], axes3[1], axes3[2], 
                            color='purple', label='F Frame Ellipse')
    
    R_T      = nu_sol.R_T
    R_T_inv  = np.linalg.inv(R_T)
    p_nu_vec = np.array([nu.px, nu.py, nu.pz])
    p_nu_F   = R_T_inv.dot(p_nu_vec)
    
    axes3[0].plot(p_nu_F[0], p_nu_F[1], 'r*', markersize=10, label='Truth Neutrino')
    axes3[1].plot(p_nu_F[0], p_nu_F[2], 'r*', markersize=10)
    axes3[2].plot(p_nu_F[1], p_nu_F[2], 'r*', markersize=10)
    
    # Add legend
    axes3[0].legend(loc='best')
    
    #plt.tight_layout()
    #plt.savefig(f'{output_prefix}_F_frame.png', dpi=150, bbox_inches='tight')
    
    # Print summary information
    print("\n" + "="*60)
    print("ELLIPSE AND DECAY ANGLE SUMMARY")
    print("="*60)
    print(f"\nMasses:")
    print(f"  Top: {top.mass:.2f} MeV")
    print(f"  W: {W.mass:.2f} MeV")
    print(f"  Neutrino: {nu.mass:.2f} MeV")
    
    print(f"\nLab Frame Momenta:")
    print(f"  Neutrino: ({nu.px:.2f}, {nu.py:.2f}, {nu.pz:.2f}) MeV")
    print(f"  W: ({W.px:.2f}, {W.py:.2f}, {W.pz:.2f}) MeV")
    
    print(f"\nDecay Angles:")
    print(f"  θ_tW (top-W in top rest): {angles['theta_tW']*180/np.pi:.1f}°")
    print(f"  θ_bW (b-W in W rest): {angles['theta_bW']*180/np.pi:.1f}°")
    print(f"  θ_lepnu (lep-nu in W rest): {angles['theta_lepnu']*180/np.pi:.1f}°")
    print(f"  θ_bW (lab): {angles['theta_bW_lab']*180/np.pi:.1f}°")
    print(f"  θ_lepnu (lab): {angles['theta_lepnu_lab']*180/np.pi:.1f}°")
    print(f"  θ_tW (lab): {angles['theta_tW_lab']*180/np.pi:.1f}°")
  
    # Ellipse parameters
    Z = nu_sol.Z
    Omega = math.sqrt(nu_sol.Om2)
    print(f"\nEllipse Parameters:")
    print(f"  Z: {Z:.2f} MeV")
    print(f"  Ω: {Omega:.2f}")
    print(f"  Z/Ω: {Z/Omega:.2f} MeV")
    print(f"  w*Z/Ω: {nu_sol.w * Z/Omega:.2f} MeV")
    plt.show()
    return angles



def something():
    import math
    
    # Given 4-momenta (GeV)
    lepton_p = [1174.21, -1031.66, -4569.23]
    bquark_p = [868.74, -667.92, -3105.45]
    E_mu, E_b = 4829.18, 3293.15
    
    # Kinematic invariants
    p_mu = math.sqrt(abs(sum(x**2 for x in lepton_p)))
    p_b = math.sqrt(abs(sum(x**2 for x in bquark_p)))
    beta_mu = p_mu / E_mu
    beta_b = p_b / E_b
    cos_theta = sum(a*b for a,b in zip(lepton_p, bquark_p)) / (p_mu * p_b)
    sin_theta = math.sqrt(abs(1 - cos_theta**2))
    theta = math.acos(cos_theta)
    
    # Half-angle
    psi = theta / 2
    
    # Cyclide parameters a, b, c from kinematics (Eqs. derived)
    a = (beta_mu + beta_b) / (2 * math.sin(psi))
    b = abs(beta_mu - beta_b) / (2 * math.cos(psi))
    c = math.sqrt(abs(a**2 - b**2))
    
    # Compute quadratic form for F^+ (physical branch)
    omega_plus = (beta_mu / beta_b - cos_theta) / sin_theta
    Omega2_plus = omega_plus**2 + (1 - beta_mu**2)
    A = (beta_mu**2 - omega_plus**2) / Omega2_plus
    B = (2 * omega_plus) / Omega2_plus
    C = -(1 - beta_mu**2) / Omega2_plus
    
    # Center of conic (solve gradient = 0)
    det = 4*A*C - B**2
    Sx0 = (-2*2*p_mu*C) / det   # from: 2A Sx0 + B Sy0 + 2p_mu = 0, B Sx0 + 2C Sy0 = 0
    Sy0 = (2*2*p_mu*B) / det
    
    # Quadratic part matrix
    Q = [[A, B/2], [B/2, C]]
    
    # Eigenvalues and eigenvectors of Q
    tr = Q[0][0] + Q[1][1]
    det_q = Q[0][0]*Q[1][1] - Q[0][1]*Q[1][0]
    lambda1 = (tr + math.sqrt(abs(tr**2 - 4*det_q))) / 2
    lambda2 = (tr - math.sqrt(abs(tr**2 - 4*det_q))) / 2
    
    # Eigenvectors (normalized)
    if abs(Q[0][1]) > 1e-12:
        v1 = [Q[0][1], lambda1 - Q[0][0]]
        v2 = [Q[0][1], lambda2 - Q[0][0]]
    else:
        v1 = [1, 0] if abs(lambda1 - Q[0][0]) < 1e-12 else [0, 1]
        v2 = [0, 1] if v1[0] == 1 else [1, 0]
    norm1 = math.sqrt(v1[0]**2 + v1[1]**2)
    norm2 = math.sqrt(v2[0]**2 + v2[1]**2)
    v1 = [v1[0]/norm1, v1[1]/norm1]
    v2 = [v2[0]/norm2, v2[1]/norm2]
    
    # Transformation from symmetric circle (u,v) to lab (Sx, Sy):
    #   [Sx' ; Sy'] = P * D * [u ; v]  where P = [v1, v2], D = diag(1/sqrt(|λ1|), 1/sqrt(|λ2|))
    #   then Sx = Sx' + Sx0, Sy = Sy' + Sy0
    D1 = 1 / math.sqrt(abs(lambda1))
    D2 = 1 / math.sqrt(abs(lambda2))
    
    # Circle radius R is determined by requiring F^+ = 0 for points on the circle.
    # For a point (u,v) on the circle: u = R cos φ, v = R sin φ.
    # In transformed coordinates: Sx' = R (v1[0] D1 cos φ + v2[0] D2 sin φ), Sy' = R (v1[1] D1 cos φ + v2[1] D2 sin φ)
    # Plug into F^+ expression (without constant term from center?) Actually, we need to include the center.
    # Let's compute R such that for φ=0, F^+ = 0.
    # For φ=0: u = R, v = 0 => Sx' = R v1[0] D1, Sy' = R v1[1] D1
    # Then Sx = Sx0 + R v1[0] D1, Sy = Sy0 + R v1[1] D1
    # Substitute into F^+ = A Sx^2 + B Sx Sy + C Sy^2 + 2 p_mu Sx + (m_mu^2 - m_nu^2) = 0
    # This is a quadratic equation in R. Solve for R.
    
    m_mu = 0.1057
    m_nu = 0.0
    E_const = m_mu**2 - m_nu**2
    
    # For φ=0, compute coefficients of R^2, R, constant
    coeff_R2 = A*(v1[0]*D1)**2 + B*(v1[0]*D1)*(v1[1]*D1) + C*(v1[1]*D1)**2
    coeff_R = 2*A*Sx0*(v1[0]*D1) + B*(Sx0*(v1[1]*D1) + Sy0*(v1[0]*D1)) + 2*C*Sy0*(v1[1]*D1) + 2*p_mu*(v1[0]*D1)
    const = A*Sx0**2 + B*Sx0*Sy0 + C*Sy0**2 + 2*p_mu*Sx0 + E_const
    
    # Solve quadratic: coeff_R2 * R^2 + coeff_R * R + const = 0
    disc = coeff_R**2 - 4*coeff_R2*const
    if disc < 0:
        R = 0
    else:
        R1 = (-coeff_R + math.sqrt(disc)) / (2*coeff_R2)
        R2 = (-coeff_R - math.sqrt(disc)) / (2*coeff_R2)
        R = R1 if R1 > 0 else R2
    
    # Now check for several φ that F^+ = 0 and compute m_W, m_t from Sx, Sy
    print("Cyclide parameters: a = {:.3f}, b = {:.3f}, c = {:.3f}".format(a, b, c))
    print("Circle radius R = {:.3f}".format(R))
    print("\nChecking points on the circle (should give constant m_W, m_t):")
    import numpy as np
    for phi in np.linspace(0, 4 * np.pi, 10000):
        u = R * math.cos(phi)
        v = R * math.sin(phi)
        Sx_prime = v1[0]*D1*u + v2[0]*D2*v
        Sy_prime = v1[1]*D1*u + v2[1]*D2*v
        Sx = Sx_prime + Sx0
        Sy = Sy_prime + Sy0
        
        # Compute F^+
        F_plus = A*Sx**2 + B*Sx*Sy + C*Sy**2 + 2*p_mu*Sx + E_const
        
        # Compute m_W and m_t from Eqs (3.1.1) and (3.1.2)
        m_W2 = -2*p_mu*Sx + m_nu**2 - m_mu**2
        m_b2 = E_b**2 - p_b**2
        m_t2 = -2*(p_b*cos_theta + p_mu)*Sx - 2*p_b*sin_theta*Sy + m_nu**2 - m_mu**2 + m_b2
        
        m_W = math.sqrt(m_W2) if m_W2 > 0 else 0
        m_t = math.sqrt(m_t2) if m_t2 > 0 else 0
        
        print("φ = {:.3f}: F^+ = {:.2e}, m_W = {:.3f} GeV, m_t = {:.3f} GeV".format(phi, F_plus, m_W, m_t))
    
    # Also compute the cyclide circle parameters from a, b, c and compare with R
    # For the cyclide containing the circle of radius R in plane z=0, we have relation:
    # R^4 - 2(a^2+b^2+c^2)R^2 + (a^2-b^2-c^2)^2 = 0
    # Check if this holds approximately.
    LHS = R**4 - 2*(a**2 + b**2 + c**2)*R**2 + (a**2 - b**2 - c**2)**2
    print("\nCyclide circle condition LHS = {:.2e} (should be 0)".format(LHS))






import sympy as sp

# Define all symbols
lam, tau, m_nu, beta_mu, p_mu, E_mu, beta_b, theta, alpha = sp.symbols(
    'lam tau m_nu beta_mu p_mu E_mu beta_b theta alpha', real=True
)


psi_p, psi_m, Omega_m, Omega_p = sp.symbols("psi{+} psi{-} o{-} o{+}", real = True)
delta = 1 - beta_mu**2

# omega^+ and omega^-
omega_p = sp.tan(psi_p) #(beta_mu/beta_b - sp.cos(theta)) / sp.sin(theta)
omega_m = sp.tan(psi_m) #(-beta_mu/beta_b - sp.cos(theta)) / sp.sin(theta)

# Omega^+ and Omega^-
#Omega_p = sp.sqrt(omega_p**2 + delta)
#Omega_m = sp.sqrt(omega_m**2 + delta)

# K^+ and K^- = sqrt(1+omega^2) = sec(psi)
K_p = sp.sqrt(1 + omega_p**2)
K_m = sp.sqrt(1 + omega_m**2)

# Centers C^+ and C^-
C_p = -E_mu/(beta_b * sp.sin(theta))
C_m =  E_mu/(beta_b * sp.sin(theta))




A, B0, B1, B2, l1 = sp.symbols("A B0 B1 B2 l", real = True)

x = 3 * beta_mu * l1 * A * sp.cosh(tau) * sp.sinh(tau) + (B0 + B1 * l1 + B2 * l1 **2)*(3 * sp.sinh(tau)**2 + 1)
print(sp.solve(x, l1))
exit()










# Define these combinations:
u_p = sp.cos(alpha) + omega_p*sp.sin(alpha)
v_p = omega_p*sp.cos(alpha) - sp.sin(alpha)   # = -sin(alpha) + omega_p*cos(alpha) ? Actually, -sin(alpha) + omega_p*cos(alpha) = v_p.

u_m = sp.cos(alpha) + omega_m*sp.sin(alpha)
v_m = omega_m*sp.cos(alpha) - sp.sin(alpha)

# We assume Z^+ = Z^- = m_nu * cosh(tau) (as before). But let's keep Z^+ and Z^- separate for now.
Z_p = m_nu * sp.cosh(tau)
Z_m = m_nu * sp.cosh(tau)  # same for now, but could be different if tau^+ != tau^-

# Compute x1 - pμ and y1 for each branch from the earlier derived expressions:
# x1 - pμ = (m_nu / K) * ( - (beta_mu/Ω) * cosh(tau) + ω * sinh(tau) )
# y1 = - (m_nu / K) * ( (ω * beta_mu / Ω) * cosh(tau) + sinh(tau) )

x1p_minus_p = (m_nu / K_p) * (- (beta_mu   / Omega_p) * sp.cosh(tau) + omega_p * sp.sinh(tau))
y1p = - (m_nu / K_p) * ((omega_p * beta_mu / Omega_p) * sp.cosh(tau) + sp.sinh(tau))

x1m_minus_p = (m_nu / K_m) * (- (beta_mu/Omega_m) * sp.cosh(tau) + omega_m * sp.sinh(tau))
y1m = - (m_nu / K_m) * ((omega_m * beta_mu / Omega_m) * sp.cosh(tau) + sp.sinh(tau))


H_p = sp.Matrix([[Z_p/Omega_p, 0, x1p_minus_p],
                 [omega_p*Z_p/Omega_p, 0, y1p],
                 [0, Z_p, 0]])

H_m = sp.Matrix([[Z_m/Omega_m, 0, x1m_minus_p],
                 [omega_m*Z_m/Omega_m, 0, y1m],
                 [0, Z_m, 0]])

# Compute the pencil matrix M = H_p - lam * H_m
M = H_p - lam * H_m

# Compute determinant P(lam, tau)
P = M.det()

# Simplify P
print("Characteristic polynomial P(λ, τ):")
P_simplified = sp.expand(P.subs(sp.sqrt(omega_p**2 +1), sp.cos(psi_p)).subs(sp.sqrt(omega_m**2 +1), sp.cos(psi_m))) #.trigsimp() #sp.simplify(P)
sp.pprint(P_simplified)

# Compute derivative dP/dτ
dP_dtau = sp.diff(P_simplified, tau).expand()
print("\nDerivative dP/dτ:")
sp.pprint(dP_dtau)

# Since we assume Z^+ = Z^- = m_nu cosh(tau), the derivative will be in terms of cosh and sinh.
# We can try to factor the derivative.
print("\nFactoring dP/dτ...")
dP_dtau_factored = sp.factor(dP_dtau)
sp.pprint(dP_dtau_factored)

# Alternatively, collect terms in cosh(tau) and sinh(tau)
from sympy import collect, cosh, sinh
dP_dtau_collected = collect(dP_dtau.expand().together(), [lam]) * Omega_p * Omega_m * sp.cos(psi_p) * sp.cos(psi_m)
dP_dtau_collected = collect(dP_dtau_collected, [cosh(tau), sinh(tau)]).subs(omega_p**2, 1 - sp.cos(psi_p)**2).subs(omega_m**2, 1 - sp.cos(psi_m)**2).expand()
sp.pprint(dP_dtau_collected)

print("\ndP/dτ collected in cosh(τ) and sinh(τ):")
sp.pprint(dP_dtau_collected)

print(sp.solve(dP_dtau_collected, lam))












