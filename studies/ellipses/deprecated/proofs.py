from sympy import *
init_printing(use_unicode=False)

# Define symbols
r, theta, delta, Sx, Sy, p_mu, m_mu, m_nu = symbols('r theta delta S_x S_y p_mu m_mu m_nu')
sin_theta, cos_theta = sin(theta), cos(theta)

# Define omega± and Omega±²
omega_plus = (r - cos_theta) / sin_theta
omega_minus = (-r - cos_theta) / sin_theta
Omega_plus_sq = omega_plus**2 + delta
Omega_minus_sq = omega_minus**2 + delta

# Construct Z^{±2} from given formula
Z_plus_sq = ((1/Omega_plus_sq - 1)*Sx**2 + 
             (2*omega_plus/Omega_plus_sq)*Sx*Sy - 
             (delta/Omega_plus_sq)*Sy**2 + 
             2*p_mu*Sx + (m_mu**2 - m_nu**2))

Z_minus_sq = ((1/Omega_minus_sq - 1)*Sx**2 + 
              (2*omega_minus/Omega_minus_sq)*Sx*Sy - 
              (delta/Omega_minus_sq)*Sy**2 + 
              2*p_mu*Sx + (m_mu**2 - m_nu**2))

# Compute ΔF = Z^{+2} - Z^{-2}
Delta_F = simplify(Z_plus_sq - Z_minus_sq)

# Extract coefficients for ΔF = α S_x² + β S_x S_y + γ S_y²
alpha_calc = Delta_F.coeff(Sx**2)
beta_calc = Delta_F.coeff(Sx*Sy)
gamma_calc = Delta_F.coeff(Sy**2)

# Expected coefficients (from problem statement)
alpha_exp = (1/Omega_plus_sq - 1/Omega_minus_sq)
beta_exp = 2*(omega_plus/Omega_plus_sq - omega_minus/Omega_minus_sq)
gamma_exp = -delta * alpha_exp

# Verify equivalence
print("α verification:", simplify(alpha_calc - alpha_exp) == 0)
print("β verification:", simplify(beta_calc - beta_exp) == 0)
print("γ verification:", simplify(gamma_calc - gamma_exp) == 0)
