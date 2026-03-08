import math

def quadratic_matrix(beta_mu, beta_b, cos_theta, sin_theta):
    # ω and Ω for the two branches
    w_plus  = omega(beta_mu, beta_b, cos_theta, sin_theta, +1)
    w_minus = omega(beta_mu, beta_b, cos_theta, sin_theta, -1)
    O_plus  = Omega(w_plus,  beta_mu)
    O_minus = Omega(w_minus, beta_mu)

    # matrix entries (Eqs. (3) in the earlier derivation)
    beta2 = beta_mu**2
    A11 = (beta2 - w_plus**2)  / (O_plus**2)
    A22 = -(beta2 - w_minus**2) / (O_minus**2)
    A12 = 0.5 * (w_plus + w_minus) * (1.0/(O_plus**2) + 1.0/(O_minus**2))

    return A11, A12, A22, w_plus, w_minus, O_plus, O_minus

def eigenvalues_2x2_symmetric(A11, A12, A22):
    """Eigenvalues of a 2×2 symmetric matrix."""
    trace = A11 + A22
    det   = A11 * A22 - A12**2
    # λ = trace/2 ± sqrt( (trace/2)² - det )
    # For a symmetric matrix this is equivalent to:
    # λ = (trace ± sqrt((A11-A22)² + 4 A12²)) / 2
    disc = (A11 - A22)**2 + 4 * A12**2
    sqrt_disc = math.sqrt(disc)
    lambda1 = (trace + sqrt_disc) / 2.0
    lambda2 = (trace - sqrt_disc) / 2.0
    return lambda1, lambda2

def rotation_angle_2x2_symmetric(A11, A12, A22):
    """
    Angle θ such that the orthogonal matrix
        R = [[cosθ, -sinθ],
             [sinθ,  cosθ]]
    diagonalises the symmetric matrix A.
    """
    # tan(2θ) = 2 A12 / (A11 - A22)
    numerator = 2.0 * A12
    denominator = A11 - A22
    if abs(denominator) < 1e-15:
        # matrix is already diagonal in the ±45° directions
        return math.copysign(math.pi/4.0, numerator)
    tan_2theta = numerator / denominator
    theta = 0.5 * math.atan(tan_2theta)
    return theta

def eigenvectors_from_angle(theta):
    """
    Eigenvectors of a 2×2 symmetric matrix given the rotation angle θ.
    The eigenvectors are the columns of the rotation matrix:
        v1 = ( cosθ, sinθ)
        v2 = (-sinθ, cosθ)
    """
    c, s = math.cos(theta), math.sin(theta)
    v1 = (c, s)
    v2 = (-s, c)
    return v1, v2

def check_diagonalisation(A11, A12, A22, theta, lambda1, lambda2):
    """Sanity check: verify that R^T A R is diagonal."""
    c, s = math.cos(theta), math.sin(theta)
    # R^T A R
    R11 = c*A11 + s*A12
    R12 = c*A12 + s*A22
    R21 = -s*A11 + c*A12
    R22 = -s*A12 + c*A22
    diag1 = c*R11 + s*R12
    diag2 = -s*R21 + c*R22
    off1  = -s*R11 + c*R12
    off2  = c*R21 + s*R22
    # off‑diagonal elements should be ~0
    return diag1, diag2, off1, off2

# --------------------------------------------------------------------
# Example using your kinematic values
# (replace these with your actual data)
beta_mu  = 0.999999989207358   # data.b_mu
beta_b   = 0.80                # data.b_b   (example – you must supply the real value)
cos_theta = math.cos(30.0 * math.pi/180.0)   # data.theta.cos
sin_theta = math.sin(30.0 * math.pi/180.0)   # data.theta.sin

# 1. Build the quadratic‑form matrix
A11, A12, A22, w_plus, w_minus, O_plus, O_minus = quadratic_matrix(
    beta_mu, beta_b, cos_theta, sin_theta
)

print("ω⁺  =", w_plus)
print("ω⁻  =", w_minus)
print("Ω⁺  =", O_plus)
print("Ω⁻  =", O_minus)
print("\nQuadratic‑form matrix A:")
print("  A11 =", A11)
print("  A12 =", A12)
print("  A22 =", A22)
print("  A = [[{:.8f}, {:.8f}],".format(A11, A12))
print("       [{:.8f}, {:.8f}]]".format(A12, A22))

# 2. Eigenvalues
lambda1, lambda2 = eigenvalues_2x2_symmetric(A11, A12, A22)
print("\nEigenvalues of A:")
print("  λ₁ =", lambda1)
print("  λ₂ =", lambda2)

# 3. Rotation angle that diagonalises A
theta = rotation_angle_2x2_symmetric(A11, A12, A22)
print("\nRotation angle θ that diagonalises A:")
print("  θ = {:.6f} rad".format(theta))
print("    = {:.6f}°".format(theta * 180.0 / math.pi))

# 4. Corresponding eigenvectors
v1, v2 = eigenvectors_from_angle(theta)
print("\nEigenvectors (unit vectors):")
print("  v₁ = ({:.6f}, {:.6f})".format(v1[0], v1[1]))
print("  v₂ = ({:.6f}, {:.6f})".format(v2[0], v2[1]))

# 5. Quick sanity check (optional)
diag1, diag2, off1, off2 = check_diagonalisation(A11, A12, A22, theta, lambda1, lambda2)
print("\nSanity check – after rotation:")
print("  diag1 ≈ λ₁ = {:.6f}".format(diag1))
print("  diag2 ≈ λ₂ = {:.6f}".format(diag2))
print("  off‑diagonal terms = {:.2e}, {:.2e}".format(off1, off2))
