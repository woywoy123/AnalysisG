import numpy as np
import math

class Particle:
    def __init__(self, px, py, pz, e):
        self.px = px; self.py = py; self.pz = pz; self.e = e
        self.p = math.sqrt(px**2 + py**2 + pz**2)
        self.m2 = max(0.0, e**2 - self.p**2)
        self.m = math.sqrt(self.m2)
        self.b = self.p / self.e

def dot4(p1, p2):
    return p1.e * p2.e - (p1.px * p2.px + p1.py * p2.py + p1.pz * p2.pz)

# ===========================================================================
# 1. KINEMATICS & INVARIANT MASSES
# ===========================================================================
# Loaded from closure testing request
lep = Particle(101034.2006, 1715.6181, 72918.7182, 124611.3672)
bqk = Particle(317335.0935, 26396.5905, 57354.7981, 323998.25)
nu  = Particle(176103.91, 98098.7484, 82607.5771, 217853.0781)

# Calculate Exact True Invariant Masses
mW2_true = lep.m2 + nu.m2 + 2 * dot4(lep, nu)
mt2_true = lep.m2 + nu.m2 + bqk.m2 + 2 * dot4(lep, nu) + 2 * dot4(lep, bqk) + 2 * dot4(nu, bqk)

print("--- 1. Extracted Physical Invariants ---")
print(f"Lepton Mass: {lep.m:.4f} MeV | Velocity (beta): {lep.b:.6f}")
print(f"b-Quark Mass: {bqk.m:.4f} MeV | Velocity (beta): {bqk.b:.6f}")
print(f"Neutrino Mass: {nu.m:.4f} MeV")
print(f"W-Boson Mass (True): {math.sqrt(mW2_true):.4f} MeV")
print(f"Top-Quark Mass (True): {math.sqrt(mt2_true):.4f} MeV")

# Invariant Mass Squared Differences (Eq. 20, 21)
Delta_MW2 = nu.m2 - lep.m2 - mW2_true
Delta_Mt2 = mW2_true + bqk.m2 - mt2_true

c_theta = (lep.px*bqk.px + lep.py*bqk.py + lep.pz*bqk.pz) / (lep.p * bqk.p)
s_theta = math.sqrt(1.0 - c_theta**2)

# ===========================================================================
# 2. REVERSE-ENGINEER TRUTH HYPERBOLIC PARAMETERS (GEOMETRIC MAP)
# ===========================================================================
# Construct the F-Frame (x1, y1, Z)
ux = np.array([lep.px/lep.p, lep.py/lep.p, lep.pz/lep.p])
b_vec = np.array([bqk.px, bqk.py, bqk.pz])
dot_b_ux = np.dot(b_vec, ux)
b_perp = b_vec - dot_b_ux * ux
uy = b_perp / np.linalg.norm(b_perp)
uz = np.cross(ux, uy)

nu_vec = np.array([nu.px, nu.py, nu.pz])
x1 = np.dot(nu_vec, ux)
y1 = np.dot(nu_vec, uy)
Z  = np.dot(nu_vec, uz)

def get_truth_params(s_omega):
    w = (1.0 / s_theta) * (s_omega * (lep.b / bqk.b) - c_theta)
    O = math.sqrt(w**2 + 1.0 - lep.b**2)
    k = math.atan(w)

    s = +1
    # The geometric mapping natively defines the physical momenta as:
    p_nu_x = -nu.m * [ s*(beta/Omega)*cosh(tau)*cos(k) + sinh(tau)*cos(phi)*sin(k) ]
    p_nu_y =  nu.m * [ sinh(tau)*cos(phi)*cos(k) - s*(beta/Omega)*cosh(tau)*sin(k) ]
    #p_nu_x = x1
    #p_nu_y = y1

    # Inverting the system to isolate sinh(tau)cos(phi) and s*cosh(tau):
    A = p_nu_y * math.cos(k) - p_nu_x * math.sin(k)
    B = -p_nu_x * math.cos(k) - p_nu_y * math.sin(k)

    psi = math.atan2(Z, A)
    if psi < 0: psi += 2 * math.pi

    val = B * O / (nu.m * lep.b)
    cosh_val = max(1.0, abs(val))
    tau_mag = math.acosh(cosh_val)

    m_nu_sinh = A / math.cos(psi) if abs(math.cos(psi)) > 0.5 else Z / math.sin(psi)
    tau = tau_mag if m_nu_sinh >= 0 else -tau_mag

    return tau, psi

tau_true_p, phi_true_p = get_truth_params(s_omega=+1)
tau_true_m, phi_true_m = get_truth_params(s_omega=-1)

# ===========================================================================
# 3. ANALYTIC MASS INVERSION (FROM DERIVATION)
# ===========================================================================
def evaluate_analytic_inversion(s_omega):
    w = (1.0 / s_theta) * (s_omega * (lep.b / bqk.b) - c_theta)
    O = math.sqrt(w**2 + 1.0 - lep.b**2)

    # Hyperbolic Boost Evaluation (Eq. 34)
    term_W = (1.0 - s_omega * (lep.b / bqk.b) * c_theta) / lep.p * Delta_MW2
    term_t = (s_omega * (lep.b / bqk.b) - c_theta) / bqk.p * Delta_Mt2
    
    bracket_tau = (1.0 / (2 * s_theta**2)) * (term_W + term_t) + (lep.p * O**2) / lep.b**2
    cosh_tau = abs((lep.b / (nu.m * O * math.sqrt(1 + w**2))) * bracket_tau)
    tau_analytic = math.acosh(cosh_tau)

    # Azimuthal Phase Evaluation (Eq. 35)
    num_phi = (1.0 / (2 * s_theta)) * (Delta_Mt2 / bqk.p - (s_omega / (bqk.b * lep.e)) * Delta_MW2) + w * lep.p
    cos_phi_analytic = num_phi / (nu.m * math.sqrt(1 + w**2) * math.sinh(tau_analytic))

    return tau_analytic, cos_phi_analytic

tau_ana_p, cosphi_ana_p = evaluate_analytic_inversion(s_omega=+1)
tau_ana_m, cosphi_ana_m = evaluate_analytic_inversion(s_omega=-1)

# ===========================================================================
# 4. CLOSURE VALIDATION
# ===========================================================================
print("\n--- 2. Closure Validation (Branch F+) ---")
print(f"TRUTH Mapping      | abs(tau): {abs(tau_true_p):12.8f} | cos(phi): {math.cos(phi_true_p):12.8f}")
print(f"ANALYTIC Inversion | abs(tau): {tau_ana_p:12.8f} | cos(phi): {cosphi_ana_p:12.8f}")
print(f"Delta              |   d(tau): {abs(abs(tau_true_p) - tau_ana_p):12.8e} | d(cosphi): {abs(math.cos(phi_true_p) - cosphi_ana_p):12.8e}")

print("\n--- 3. Closure Validation (Branch F-) ---")
print(f"TRUTH Mapping      | abs(tau): {abs(tau_true_m):12.8f} | cos(phi): {math.cos(phi_true_m):12.8f}")
print(f"ANALYTIC Inversion | abs(tau): {tau_ana_m:12.8f} | cos(phi): {cosphi_ana_m:12.8f}")
print(f"Delta              |   d(tau): {abs(abs(tau_true_m) - tau_ana_m):12.8e} | d(cosphi): {abs(math.cos(phi_true_m) - cosphi_ana_m):12.8e}")
