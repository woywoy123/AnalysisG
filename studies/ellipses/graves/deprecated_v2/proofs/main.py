from tests import *

#test_mW2()
#test_mT2()
#test_Z2()
#test_deltaG2()
#test_relations()
#test_deltaG2R()
#test_rotation()
#test_eigenvalues()
#test_diagonalization()
#test_kappa()
#test_mobius()

import sympy as sp
Sx, Sy, w, theta = symbols(["Sx", "Sy", "w", "theta"])
mu, nu, b, W = particle("mu"), particle("nu"), particle("b"), particle("W")

wp, wm, op, om, theta, m_nu = symbols(["w^{+}", "w^{-}", "o^{+}", "o^{-}", "theta", "m_nu"])
mu, b = particle("mu"), particle("b")


wp_  =  1 / sp.sin(theta) * (mu.beta/b.beta - sp.cos(theta))
wm_  = -1 / sp.sin(theta) * (mu.beta/b.beta + sp.cos(theta))
op2_ = wp**2 + 1 - mu.beta ** 2
om2_ = wm**2 + 1 - mu.beta ** 2

z2p = test_Z2().subs(w, wp)
z2m = test_Z2().subs(w, wm)
G2t = (z2p - z2m)
G2t = sp.expand(G2t).together()

G2c = (1 / op2_ - 1 / om2_) * Sx ** 2 + 2 * ( wp / op2_ - wm / om2_) * Sx * Sy + ( wp**2 / op2_ - wm**2 / om2_) * Sy **2
G2c = sp.expand(G2c).together()

gamma = (1 - mu.beta**2)
G2c = 1 / (op2_ * om2_) * ( (om2_ - op2_) * Sx ** 2 + 2 * ( om2_ * wp - op2_ * wm ) * Sx * Sy + (wp ** 2 * om2_ - wm ** 2 * op2_) * Sy **2 )
proof(G2t, G2c, "Test Delta G^2")

G2c = 1 / (op2_ * om2_) * ( (om2_ - op2_) * Sx ** 2 + 2 * ( om2_ * wp - op2_ * wm ) * Sx * Sy - gamma * (om2_ - op2_) * Sy ** 2 ) 
proof(G2t, G2c, "Test Delta G^2 Factor")

G2c = (om2_ - op2_) / (op2_ * om2_) * (Sx ** 2 + 2 * ( om2_ * wp - op2_ * wm ) * Sx * Sy / (om2_ - op2_) - gamma * Sy ** 2 ) 
proof(G2t, G2c, "Test Delta G^2 Factor")

G2c = (om2_ - op2_) / (op2_ * om2_) * (Sx ** 2 + 2 * ( om2_ * wp - op2_ * wm ) * Sx * Sy / (wm**2 - wp**2) - gamma * Sy ** 2 ) 
proof(G2t, G2c, "Test Delta G^2 Factor")

ap = (wm + wp)
am = (wm - wp)
G2c = (om2_ - op2_) / (op2_ * om2_) * (Sx ** 2 + 2 * ( wm * wp * am - gamma * am) * Sx * Sy / (ap * am) - gamma * Sy ** 2 ) 
proof(G2t, G2c, "Test Delta G^2 Factor")

G2c = (om2_ - op2_) / (op2_ * om2_) * ( Sx ** 2 + 2 * ( wm * wp - gamma) * Sx * Sy / (ap) - gamma * Sy ** 2 ) 
proof(G2t, G2c, "Test Delta G^2 Factor")

op_ = sp.sqrt(wp**2 + 1 - mu.beta ** 2)
om_ = sp.sqrt(wm**2 + 1 - mu.beta ** 2)

lb1 = ((op_ - om_) ** 2 - (wp + wm)**2) / (2 * (wp + wm))
lb2 = ((op_ + om_) ** 2 - (wp + wm)**2) / (2 * (wp + wm))

Gc = ((om2_ - op2_) / sp.sqrt(om2_ * op2_)) * ( Sx - lb1 * Sy )*(Sx - lb2 * Sy)
proof(G2t, G2c, "Test Delta G^2 Factor in terms of eigenvalues")

G2c = -gamma
proof((lb1 * lb2).subs(wm, wm_).subs(wp, wp_), G2c, "lambda1 x lambda2")

G2c = 2 * ( (gamma - wp * wm) / (wp + wm) )
proof(lb1 + lb2, G2c, "lambda1 + lambda2")


M = sp.Matrix([[1, - (lb1 + lb2)/2], [- (lb1 + lb2)/2, -gamma]])
tpsit = 2 * M[1] / (M[0] - M[3])

tpsi_ = - (lb2 + lb1) / (1 - lb1*lb2)
proof(tpsit, tpsi_, "tan(2psi)")

lb1 = ((op_ - om_) ** 2 - (wp + wm)**2) / (2 * (wp + wm))
lb2 = ((op_ + om_) ** 2 - (wp + wm)**2) / (2 * (wp + wm))

#alpha_kpp = 1 / op_**2 - 1 + 2 * wp * lb1 / op_**2 + ((wp / op_)**2 - 1) * lb1 **2
#alpha_kpp = sp.expand(alpha_kpp).together()
#alpha_kpp = alpha_kpp.subs(wp, wp_).expand()


op2_ = - mu.beta ** 2 + wp**2 + 1 
om2_ = - mu.beta ** 2 + wm**2 + 1 



kp1, km1, s = symbols(["kappa_{+}", "kappa_{-}", "s"])

z2p = sp.simplify(test_Z2().subs(w, wp).subs(Sy, kp1 * Sx)).collect(Sx)
z2m = sp.simplify(test_Z2().subs(w, wm).subs(Sy, km1 * Sx)).collect(Sx)
Sxpkp, Sxmkp = sp.solve(z2p, Sx)
sxpkm, Sxmkm = sp.solve(z2m, Sx)

s1 = Sxpkp + Sxmkp
s2 = Sxpkp * Sxmkp
s1_s2 = s2 / s1

s1c = (2 * mu.p) / (1 + kp1 * sp.tan(theta))
sx = (s1 - s1c).subs(wm, wm_).subs(wp, wp_).trigsimp().expand()
print(sx.trigsimp().collect(s1c))
sp.pprint(sp.solve(sx, s1c))
#print(sp.simplify(sp.expand(s2.subs(wm, wm_).subs(wp, wp_) )))
#print(sp.simplify(s1_s2))





#print(Sxpkp)















exit()







import sympy as sp

# Define symbols
p_mu, p_b, m_mu, m_b, m_nu, theta = sp.symbols('p_mu p_b m_mu m_b m_nu theta', positive=True)
Sx, Sy = sp.symbols('Sx Sy')

# Define derived quantities
Delta = p_mu * p_b * sp.cos(theta)           # dot product
delta = p_mu * p_b * sp.sin(theta)           # |cross product| magnitude
alpha = p_b**2 * m_mu**2 - p_mu**2 * m_b**2  # defined invariant

E_mu = sp.sqrt(p_mu**2 + m_mu**2)
E_b  = sp.sqrt(p_b**2 + m_b**2)
beta_mu = p_mu / E_mu
beta_b  = p_b / E_b

gamma = 1 - beta_mu**2

# Define omega_+, omega_-
sin_theta = sp.sin(theta)
cos_theta = sp.cos(theta)
omega_p = (beta_mu/beta_b - cos_theta) / sin_theta
omega_m = (-beta_mu/beta_b - cos_theta) / sin_theta

Omega_p_sq = omega_p**2 + gamma
Omega_m_sq = omega_m**2 + gamma

# Coefficients for Z^2_+
A_p = (beta_mu**2 - omega_p**2) / Omega_p_sq
B_p = (2*omega_p) / Omega_p_sq
C_p = -gamma / Omega_p_sq

# Coefficients for Z^2_-
A_m = (beta_mu**2 - omega_m**2) / Omega_m_sq
B_m = (2*omega_m) / Omega_m_sq
C_m = -gamma / Omega_m_sq

# Common linear and constant terms
D = 2 * p_mu
E = m_mu**2 - m_nu**2

# Quadratic forms
Z2_p = A_p*Sx**2 + B_p*Sx*Sy + C_p*Sy**2 + D*Sx + E
Z2_m = A_m*Sx**2 + B_m*Sx*Sy + C_m*Sy**2 + D*Sx + E

# Proposed expression for S_x at intersection points
S_x_proposed_p = (-Delta + sp.sqrt(Delta**2 + delta**2 - alpha + m_nu**2*(p_mu**2 + m_mu**2))) / (2*p_mu)
S_x_proposed_m = (-Delta - sp.sqrt(Delta**2 + delta**2 - alpha + m_nu**2*(p_mu**2 + m_mu**2))) / (2*p_mu)

Sxp = sp.simplify(Z2_p.subs(Sy, Sx * lb1).collect(Sx))
r1, r2 = sp.solve(Sxp, Sx)

print(r1)
print(r2)











