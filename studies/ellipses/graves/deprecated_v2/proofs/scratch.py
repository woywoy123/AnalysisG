import sympy as sp
import cmath

# Define symbols
beta_mu, beta_b, theta, Sx, Sy = sp.symbols('beta_mu beta_b theta Sx Sy', real=True)
# Auxiliary symbols
A = beta_mu/(beta_b*sp.sin(theta))
B = sp.cos(theta)/sp.sin(theta)  # cot(theta)
C = 1 - beta_mu**2  # gamma_mu^{-2}
R = A**2 + B**2 + C

# omega definitions
omega_plus = A - B
omega_minus = -A - B
# Omega definitions
Omega_plus_sq = omega_plus**2 + C
Omega_minus_sq = omega_minus**2 + C

# Check R ± 2AB equals Omega_±^2
print("=== Step 1: Check Omega_±^2 == R ∓ 2AB ===")
print("Omega_plus^2 - (R - 2*A*B) =", sp.simplify(Omega_plus_sq - (R - 2*A*B)))
print("Omega_minus^2 - (R + 2*A*B) =", sp.simplify(Omega_minus_sq - (R + 2*A*B)))
print()

# Coefficients a, d, c for ΔG²
a = (beta_mu**2 - omega_plus**2)/Omega_plus_sq - (beta_mu**2 - omega_minus**2)/Omega_minus_sq
d = 2*omega_plus/Omega_plus_sq - 2*omega_minus/Omega_minus_sq
c = -C*(1/Omega_plus_sq - 1/Omega_minus_sq)

print("=== Step 2: Coefficients a, d, c ===")
print("a (simplified) =", sp.simplify(a))
print("Expected: 4*A*B/(Ω_+^2 Ω_-^2)")
print("Check: a - 4*A*B/(Omega_plus_sq*Omega_minus_sq) =", 
      sp.simplify(a - 4*A*B/(Omega_plus_sq*Omega_minus_sq)))
print()

print("d (simplified) =", sp.simplify(d))
print("Expected: 4*A*(R - 2*B^2)/(Ω_+^2 Ω_-^2)")
print("Check: d - 4*A*(R - 2*B**2)/(Omega_plus_sq*Omega_minus_sq) =", 
      sp.simplify(d - 4*A*(R - 2*B**2)/(Omega_plus_sq*Omega_minus_sq)))
print()

print("c (simplified) =", sp.simplify(c))
print("Expected: -4*A*B*C/(Ω_+^2 Ω_-^2)")
print("Check: c + 4*A*B*C/(Omega_plus_sq*Omega_minus_sq) =", 
      sp.simplify(c + 4*A*B*C/(Omega_plus_sq*Omega_minus_sq)))
print()

# ΔG² from coefficients
DeltaG2 = a*Sx**2 + d*Sx*Sy + c*Sy**2

# Factorization
K = 4*A*B/(Omega_plus_sq*Omega_minus_sq)
# Quadratic form inside brackets: Sx^2 + ((R-2*B^2)/B)*Sx*Sy - C*Sy^2
# But note: K * [Sx^2 + ((R-2*B^2)/B)*Sx*Sy - C*Sy^2] should equal DeltaG2
DeltaG2_factor = K * (Sx**2 + (R - 2*B**2)/B * Sx*Sy - C*Sy**2)

print("=== Step 3: Factorization of ΔG² ===")
print("ΔG² from coefficients - ΔG² from factor form =", 
      sp.simplify(DeltaG2 - DeltaG2_factor))
print()

# Now, find λ± from the quadratic equation:
# λ^2 + ((R-2*B^2)/B) * λ - C = 0
lambda_sym = sp.symbols('lambda')
quad_eq = lambda_sym**2 + (R - 2*B**2)/B * lambda_sym - C
# Solve for λ
lambda_solutions = sp.solve(quad_eq, lambda_sym)
lambda_plus, lambda_minus = lambda_solutions[0], lambda_solutions[1]

print("λ_+ =", sp.simplify(lambda_plus))
print("λ_- =", sp.simplify(lambda_minus))
print()

# Check that (Sx - λ- Sy)(Sx - λ+ Sy) gives the same quadratic form (up to a factor)
# Actually, the quadratic form in the factorization is:
# (Sx - λ- Sy)(Sx - λ+ Sy) = Sx^2 - (λ+ + λ-) Sx Sy + λ+ λ- Sy^2
# We know from the quadratic equation that:
# λ+ + λ- = -(R-2*B^2)/B and λ+ λ- = -C
# So indeed the quadratic form becomes: Sx^2 + ((R-2*B^2)/B) Sx Sy - C Sy^2
# Therefore, the factorization is correct.

# Check the product and sum:
print("λ_+ + λ_- =", sp.simplify(lambda_plus + lambda_minus))
print("Expected: -(R-2*B^2)/B =", sp.simplify(-(R-2*B**2)/B))
print("Check difference:", sp.simplify(lambda_plus + lambda_minus + (R-2*B**2)/B))
print()

print("λ_+ * λ_- =", sp.simplify(lambda_plus * lambda_minus))
print("Expected: -C =", -C)
print("Check difference:", sp.simplify(lambda_plus * lambda_minus + C))
print()

# Now, compute the eigenvalues of matrix M
M = sp.Matrix([[a, d/2], [d/2, c]])
print("=== Step 4: Eigenvalues of matrix M ===")
# Compute eigenvalues symbolically
eigenvals = M.eigenvals()
# The eigenvalues are the roots of the characteristic polynomial
# Let's compute them via the characteristic polynomial
mu = sp.symbols('mu')
char_poly = M.charpoly(mu)
print("Characteristic polynomial:", sp.simplify(char_poly.as_expr()))
print()

# Compute eigenvalues directly
eigenvals_list = M.eigenvects()
for eig in eigenvals_list:
    print("Eigenvalue:", sp.simplify(eig[0]))
    # The expression is too complicated. Let's try to simplify it.
    # We'll compute the simplified eigenvalue expression from the formula:
    # μ = (a+c)/2 ± sqrt((a+c)^2/4 - (ac - d^2/4))
    a_expr = a
    c_expr = c
    d_expr = d
    # Compute (a+c)/2
    half_sum = (a_expr + c_expr)/2
    # Compute discriminant inside sqrt: (a+c)^2/4 - (ac - d^2/4)
    disc_inside = half_sum**2 - (a_expr*c_expr - d_expr**2/4)
    # Simplify disc_inside
    disc_inside_simplified = sp.simplify(disc_inside)
    print("Discriminant inside sqrt (simplified):", disc_inside_simplified)
    # Now, the eigenvalues are: μ = half_sum ± sqrt(disc_inside)
    # Let's check if disc_inside is a perfect square in terms of Ω's and A, B, etc.
    # We know from the theory that disc_inside should be: (4A^2)/(Ω_+^2 Ω_-^2) + (4A^2B^2 βμ^4)/(Ω_+^4 Ω_-^4)
    # But note: the expression we have for disc_inside might simplify to that.
    # Let's express disc_inside in terms of A, B, C, R, Ω's.
    # We already have a, c, d in terms of these.
    # We'll substitute the expressions for a, c, d in terms of A, B, etc.
    # Actually, we can compute disc_inside using the formulas for a, c, d.
    # Alternatively, compute directly:
    # half_sum = (a+c)/2 = (4AB/(Ω+²Ω-²) * (βμ² - C) )? Actually, from the theory:
    # a+c = (4ABβμ²)/(Ω+²Ω-²) because a = 4AB/(Ω+²Ω-²) and c = -4ABC/(Ω+²Ω-²), so a+c = 4AB(1-C)/(Ω+²Ω-²) = 4ABβμ²/(Ω+²Ω-²)
    # So half_sum = 2ABβμ²/(Ω+²Ω-²)
    # And ac - d^2/4 = -4A^2/(Ω+²Ω-²) from the theory.
    # So disc_inside = (2ABβμ²/(Ω+²Ω-²))^2 + 4A^2/(Ω+²Ω-²) = 4A^2/(Ω+²Ω-²) * ( (B^2 βμ^4)/(Ω+²Ω-²) + 1 )
    # But note: Ω+²Ω-² = R^2 - 4A^2B^2.
    # Let's verify:
    half_sum_theory = 2*A*B*beta_mu**2/(Omega_plus_sq*Omega_minus_sq)
    disc_inside_theory = half_sum_theory**2 + 4*A**2/(Omega_plus_sq*Omega_minus_sq)
    print("Half sum (theory):", sp.simplify(half_sum_theory))
    print("Half sum (from a+c):", sp.simplify(half_sum))
    print("Difference in half sum:", sp.simplify(half_sum - half_sum_theory))
    print()
    print("Disc inside (theory):", sp.simplify(disc_inside_theory))
    print("Disc inside (computed):", sp.simplify(disc_inside))
    print("Difference in disc inside:", sp.simplify(disc_inside - disc_inside_theory))
    print()
    # Now, the eigenvalues should be: half_sum ± sqrt(disc_inside_theory)
    # Let's compute the square root of disc_inside_theory and see if it matches the messy expression from eigenvals.
    sqrt_disc = sp.sqrt(disc_inside_theory)
    # We can try to simplify sqrt_disc by substituting the expressions for Omega_plus_sq and Omega_minus_sq.
    # But note: disc_inside_theory = 4A^2/(Ω+²Ω-²) * (1 + (B^2 βμ^4)/(Ω+²Ω-²))
    # We can also note that Ω+²Ω-² = (R^2 - 4A^2B^2)
    # Let's compute the eigenvalue from the formula and compare with one of the eigenvalues from M.eigenvals().
    mu1 = half_sum_theory + sqrt_disc
    mu2 = half_sum_theory - sqrt_disc
    print("Eigenvalue from formula (μ1):", sp.simplify(mu1))
    print("Eigenvalue from formula (μ2):", sp.simplify(mu2))
    # Now, let's get the eigenvalues from M.eigenvals() and compare.
    # Since eigenvals is a dictionary, we can extract the eigenvalues.
    M_eigenvals = list(eigenvals.keys())
    print("Eigenvalues from M.eigenvals():")
    for ev in M_eigenvals:
        print(ev)
    # Compare the first eigenvalue from M with mu1 and mu2.
    # We'll compute the difference between mu1 and the first eigenvalue from M.
    diff1 = sp.simplify(mu1 - M_eigenvals[0])
    diff2 = sp.simplify(mu1 - M_eigenvals[1])
    print("Difference between mu1 and first eigenvalue from M:", diff1)
    print("Difference between mu1 and second eigenvalue from M:", diff2)
    # If one of the differences is zero, then they match.
    break  # We only need one iteration because we are not using the eigenvects for anything else.
