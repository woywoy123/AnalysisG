import sympy as sp

# Define symbols
Sx, Sy, m_nu = sp.symbols('Sx Sy m_nu', real=True)
# Measured quantities
p_mu, E_mu, p_b, E_b = sp.symbols('p_mu E_mu p_b E_b', real=True, positive=True)
c, s = sp.symbols('c s', real=True)  # cosθ and sinθ
m_mu, m_b = sp.symbols('m_mu m_b', real=True, positive=True)  # masses of muon and b quark

# Derived quantities
beta_mu = p_mu / E_mu
beta_b = p_b / E_b
a = 1 - beta_mu**2  # γ_μ^{-2}
u = Sx * beta_mu**2

# Expressions for m_W^2 and m_t^2 from earlier (in terms of Sx, Sy, m_nu)
mW_sq = m_nu**2 - m_mu**2 - 2 * p_mu * Sx
# x0' = β_b * (c*Sx + s*Sy)
x0_prime = beta_b * (c*Sx + s*Sy)

# For A_μ: d_μ = m_W^2 - x0^2 - ε^2
# x0 = β_μ * Sx  (from earlier derivation)
x0 = beta_mu * Sx
epsilon_sq = a * (mW_sq - m_nu**2)
d_mu = mW_sq - x0**2 - epsilon_sq

# For A_b: 
# v1 = c * x0' * β_b = c * β_b * (β_b*(c*Sx+s*Sy)) = β_b**2 * c * (c*Sx+s*Sy)
# v2 = s * x0' * β_b = β_b**2 * s * (c*Sx+s*Sy)
v1 = beta_b**2 * c * (c*Sx + s*Sy)
v2 = beta_b**2 * s * (c*Sx + s*Sy)
# d_b = m_W^2 - x0'^2
d_b = mW_sq - x0_prime**2

# Build matrices A_mu and A_b
# Using homogeneous coordinates (x, y, z, 1)
A_mu = sp.Matrix([
    [a, 0, 0, u],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [u, 0, 0, d_mu]
])

A_b = sp.Matrix([
    [1 - (c*beta_b)**2, -c*s*beta_b**2, 0, v1],
    [-c*s*beta_b**2, 1 - (s*beta_b)**2, 0, v2],
    [0, 0, 1, 0],
    [v1, v2, 0, d_b]
])

# Define λ
lam = sp.symbols('lambda')

# Form the pencil P(λ) = A_mu - λ * A_b
P = A_mu - lam * A_b

print(P.det())

# Compute determinant of P
det_P = sp.simplify(P.det())

# Factor the determinant
det_P_factored = sp.factor(det_P)

# We expect (1-λ)^2 * Q(λ)
# Extract Q(λ) by dividing det_P by (1-λ)^2
Q = sp.simplify(det_P_factored / (1 - lam)**2)

# Now, Q should be a quadratic in λ
# Expand Q to see coefficients
Q_expanded = sp.expand(Q)

# Display results
print("Determinant of the pencil (factored):")
print(det_P_factored)
print("\nQuadratic Q(λ) after dividing by (1-λ)^2:")
print(Q_expanded)
print("\nQ(λ) expressed as A*λ^2 + B*λ + C:")
# Collect terms in λ
Q_poly = sp.Poly(Q_expanded, lam)
coefficients = Q_poly.all_coeffs()
A, B, C = coefficients
print(f"A = {A}")
print(f"B = {B}")
print(f"C = {C}")
