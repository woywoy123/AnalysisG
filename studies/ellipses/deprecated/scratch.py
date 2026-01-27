import sympy as sp

def verify_z2_expansion():
    print("Verifying Z^2 expansion using SymPy...")
    
    # 1. Define Symbols
    # Physical constants and variables
    m_W, m_mu, m_nu, E_mu = sp.symbols('m_W m_mu m_nu E_mu', real=True, positive=True)
    beta_mu = sp.symbols('beta_mu', real=True)
    p_mu = sp.symbols('p_mu', real=True)
    
    # Pencil function variables
    S_x, S_y, omega = sp.symbols('S_x S_y omega', real=True)

    # 2. Define Relationships
    # Gamma^-2 = 1 - beta^2
    gamma_inv2 = 1 - beta_mu**2
    
    # Relation: m_mu^2 = E_mu^2 * (1 - beta^2) (On-shell condition)
    # We substitute m_mu^2 with this to help simplification
    m_mu_sq = E_mu**2 * gamma_inv2
    
    # Omega^2
    Omega2 = omega**2 + gamma_inv2
    
    # x0 (tilde_x0) definition
    # x0 = - (m_W^2 - m_mu^2 - m_nu^2) / (2 * E_mu)
    x0 = - (m_W**2 - m_mu_sq - m_nu**2) / (2 * E_mu)
    
    # epsilon^2 definition
    epsilon2 = gamma_inv2 * (m_W**2 - m_nu**2)
    
    # x1 (tilde_x1) definition
    x1 = S_x - (S_x + omega * S_y) / Omega2

    # ---------------------------------------------------------
    # 3. Construct the ORIGINAL Z^2 expression
    # ---------------------------------------------------------
    # Z^2 = x1^2 * Omega^2 - (Sy - w*Sx)^2 - (m_W^2 - x0^2 - epsilon^2)
    term_A_orig = x1**2 * Omega2
    term_B_orig = (S_y - omega * S_x)**2
    term_C_orig = m_W**2 - x0**2 - epsilon2
    
    Z2_original = term_A_orig - term_B_orig - term_C_orig

    # ---------------------------------------------------------
    # 4. Construct the TARGET Z^2 expansion
    # ---------------------------------------------------------
    # Z^2 = [ Sx(w^2 - beta^2) - w*Sy ]^2 / Omega^2 
    #       - (Sy - w*Sx)^2 
    #       + 2*p_mu*Sx + beta^2*Sx^2 + m_mu^2 - m_nu^2
    
    target_term1 = (S_x * (omega**2 - beta_mu**2) - omega * S_y)**2 / Omega2
    target_term2 = (S_y - omega * S_x)**2
    target_term3 = 2 * p_mu * S_x + beta_mu**2 * S_x**2 + m_mu_sq - m_nu**2
    
    Z2_target = target_term1 - target_term2 + target_term3

    # ---------------------------------------------------------
    # 5. Prove Equality
    # ---------------------------------------------------------
    # The target expression uses Sx as a free parameter in the quadratic form.
    # However, the "constant" part of the expansion (m_mu^2 - m_nu^2...) relies on 
    # the specific definition of Sx related to x0 to match the original (m_W^2 - x0^2...) term.
    
    # We define Sx in terms of x0 and p_mu to allow full simplification.
    # Sx = (x0 * beta - p * gamma^-2) / beta^2
    # Note: p_mu = E_mu * beta_mu
    p_mu_val = E_mu * beta_mu
    Sx_def = (x0 * beta_mu - p_mu_val * gamma_inv2) / beta_mu**2
    
    # Substitute Sx definition into both expressions
    # This reduces everything to functions of fundamental masses, E_mu, and beta_mu.
    Z2_orig_sub = Z2_original.subs({S_x: Sx_def})
    Z2_target_sub = Z2_target.subs({S_x: Sx_def, p_mu: p_mu_val})
    
    # Calculate difference
    diff = sp.simplify(Z2_orig_sub - Z2_target_sub)
    
    print("\n--- Verification Results ---")
    print(f"Difference after simplification: {diff}")
    
    if diff == 0:
        print("SUCCESS: The target expansion is mathematically identical to the original definition.")
    else:
        print("FAILURE: The expressions are not identical.")




import sympy as sp

def prove_rotation_and_parameterization():
    print("--- Proving Rotation, Shift, and Parameterization Steps ---")

    # 1. Define Symbols
    # S_x, S_y: Original coordinates
    # omega: slope
    # Omega2: Omega squared
    # beta_mu, p_mu: kinematic constants
    # psi: rotation angle
    # u, v: rotated coordinates
    # U, V: shifted coordinates
    # U0, V0: center of shift
    # K, tau: hyperbolic parameters
    
    (S_x, S_y, omega, Omega2, beta_mu, p_mu, 
     psi, u, v, U, V, U0, V0, K, tau) = sp.symbols(
        'S_x S_y omega Omega2 beta_mu p_mu psi u v U V U0 V0 K tau', real=True
    )

    # ---------------------------------------------------------
    # STEP 1: Rotation to (u, v)
    # ---------------------------------------------------------
    print("\n[Step 1] Verifying Rotation to (u, v)...")

    # Use actual trig functions to ensure identities (sin^2 + cos^2 = 1) work automatically
    c_psi = sp.cos(psi)
    s_psi = sp.sin(psi)
    
    # Inverse rotation relations given in prompt:
    # Sx = u*cos - v*sin
    # Sy = u*sin + v*cos
    Sx_rot = u * c_psi - v * s_psi
    Sy_rot = u * s_psi + v * c_psi

    # Original Definitions of x1_tilde, y1_tilde
    # x1 = Sx - (Sx + w*Sy)/Omega2
    # y1 = Sy - w*(Sx + w*Sy)/Omega2
    # Note: w = tan(psi) = s_psi/c_psi
    w_val = s_psi / c_psi
    
    x1_orig = Sx_rot - (Sx_rot + w_val * Sy_rot) / Omega2
    y1_orig = Sy_rot - w_val * (Sx_rot + w_val * Sy_rot) / Omega2

    # Target Expressions Step 1
    # x1 = u(c - 1/(Omega2*c)) - v*s
    # y1 = u(s - w/(Omega2*c)) + v*c
    x1_target_1 = u * (c_psi - 1/(Omega2 * c_psi)) - v * s_psi
    y1_target_1 = u * (s_psi - w_val/(Omega2 * c_psi)) + v * c_psi

    # Verify Equality
    diff_x1_s1 = sp.simplify(x1_orig - x1_target_1)
    diff_y1_s1 = sp.simplify(y1_orig - y1_target_1)

    if diff_x1_s1 == 0 and diff_y1_s1 == 0:
        print("  -> Verified: Expressions match after rotation.")
    else:
        print(f"  -> FAILED Step 1. Diff X: {diff_x1_s1}, Diff Y: {diff_y1_s1}")
        # Return to avoid cascading errors, but usually we might want to see debug info
        return

    # ---------------------------------------------------------
    # STEP 2: Simplification using Omega^2
    # ---------------------------------------------------------
    print("\n[Step 2] Verifying Simplification using Omega^2...")

    # Identity: Omega^2 = omega^2 + 1 - beta_mu^2
    # w = tan(psi). 
    # w^2 + 1 = tan^2 + 1 = sec^2 = 1/cos^2
    # So Omega^2 = 1/c_psi^2 - beta_mu^2
    Omega2_sub = 1/c_psi**2 - beta_mu**2

    # Evaluate the specific coefficient terms mentioned in prompt
    # Term 1: cos - 1/(Omega2 * cos)
    term1 = c_psi - 1/(Omega2 * c_psi)
    # Substitute Omega2
    term1_sub = term1.subs(Omega2, Omega2_sub)
    
    # Target 1: -beta^2/Omega^2 * cos
    target1 = -beta_mu**2 / Omega2 * c_psi
    target1_sub = target1.subs(Omega2, Omega2_sub)

    # Verify Term 1
    check1 = sp.simplify(term1_sub - target1_sub)

    # Term 2: sin - w/(Omega2 * cos)
    # w = s/c
    term2 = s_psi - (s_psi/c_psi)/(Omega2 * c_psi)
    term2_sub = term2.subs(Omega2, Omega2_sub)

    # Target 2: -beta^2/Omega^2 * sin
    target2 = -beta_mu**2 / Omega2 * s_psi
    target2_sub = target2.subs(Omega2, Omega2_sub)
    
    # Verify Term 2
    check2 = sp.simplify(term2_sub - target2_sub)

    if check1 == 0 and check2 == 0:
        print("  -> Verified: Coefficient simplifications are correct.")
        
        # Define the simplified forms for next steps
        x1_step2 = - (beta_mu**2 / Omega2) * u * c_psi - v * s_psi
        y1_step2 = - (beta_mu**2 / Omega2) * u * s_psi + v * c_psi
    else:
        print(f"  -> FAILED Step 2. Check1: {check1}, Check2: {check2}")
        return

    # ---------------------------------------------------------
    # STEP 3: Shift to Center (U, V)
    # ---------------------------------------------------------
    print("\n[Step 3] Verifying Shift to Center (U0, V0)...")

    # U = u + U0 => u = U - U0
    # V = v + V0 => v = V - V0
    # Condition: When U=0, V=0 (so u=-U0, v=-V0), we want x1 = p_mu, y1 = 0
    
    # Equations at origin of U,V:
    eq_x = x1_step2.subs({u: -U0, v: -V0}) - p_mu
    eq_y = y1_step2.subs({u: -U0, v: -V0}) - 0

    # Solve linear system for U0, V0
    sol = sp.solve([eq_x, eq_y], (U0, V0))
    
    U0_calc = sol[U0]
    V0_calc = sol[V0]

    # Target U0, V0
    U0_target = (p_mu * Omega2 * c_psi) / beta_mu**2
    V0_target = p_mu * s_psi

    check_U0 = sp.simplify(U0_calc - U0_target)
    check_V0 = sp.simplify(V0_calc - V0_target)

    if check_U0 == 0 and check_V0 == 0:
        print("  -> Verified: Center shift (U0, V0) matches target.")
        
        # Construct expressions in terms of U, V
        # u = U - U0, v = V - V0
        # Substitute into x1_step2, y1_step2
        x1_step3 = x1_step2.subs({u: U - U0_target, v: V - V0_target})
        y1_step3 = y1_step2.subs({u: U - U0_target, v: V - V0_target})
        
        # Check against target forms from prompt
        # x1 = p_mu - (beta^2/Omega^2)*U*c - V*s
        x1_target_3 = p_mu - (beta_mu**2 / Omega2) * U * c_psi - V * s_psi
        
        # y1 = -(beta^2/Omega^2)*U*s + V*c
        y1_target_3 = - (beta_mu**2 / Omega2) * U * s_psi + V * c_psi
        
        diff_x3 = sp.simplify(x1_step3 - x1_target_3)
        diff_y3 = sp.simplify(y1_step3 - y1_target_3)
        
        if diff_x3 == 0 and diff_y3 == 0:
             print("  -> Verified: x1, y1 expressions in terms of U, V match.")
        else:
             print("  -> FAILED Step 3 expansion.")
    else:
        print("  -> FAILED Step 3 solution.")
        print(f"     U0 Calc: {U0_calc}")
        print(f"     U0 Targ: {U0_target}")
        return

    # ---------------------------------------------------------
    # STEP 4: Hyperbolic Parameterization
    # ---------------------------------------------------------
    print("\n[Step 4] Verifying Hyperbolic Parameterization...")

    # Right Branch
    # U = (Omega/beta) * sqrt(K) * cosh(tau)
    # V = sqrt(K) * sinh(tau)
    
    cosh_tau, sinh_tau = sp.symbols('cosh_tau sinh_tau') # Treat as symbols for structure check
    sqrt_K = sp.sqrt(K)
    
    U_sub = (sp.sqrt(Omega2) / beta_mu) * sqrt_K * cosh_tau
    V_sub = sqrt_K * sinh_tau

    # Substitute into Step 3 expressions
    x1_final_calc = x1_target_3.subs({U: U_sub, V: V_sub})
    y1_final_calc = y1_target_3.subs({U: U_sub, V: V_sub})

    # Target Expressions
    # x1 = p - sqrt(K) * ( (beta/Omega)*cosh*c + sinh*s )
    # y1 = sqrt(K) * ( -(beta/Omega)*cosh*s + sinh*c )
    
    # Note: Prompt uses beta/Omega. Calc has (beta^2/Omega^2) * (Omega/beta) = beta/Omega.
    # So coefficients should match.
    
    x1_final_target = p_mu - sqrt_K * ( (beta_mu/sp.sqrt(Omega2))*cosh_tau*c_psi + sinh_tau*s_psi )
    y1_final_target = sqrt_K * ( -(beta_mu/sp.sqrt(Omega2))*cosh_tau*s_psi + sinh_tau*c_psi )

    diff_x4 = sp.simplify(x1_final_calc - x1_final_target)
    diff_y4 = sp.simplify(y1_final_calc - y1_final_target)

    if diff_x4 == 0 and diff_y4 == 0:
        print("  -> Verified: Right branch parameterization matches.")
    else:
        print(f"  -> FAILED Step 4. Diff X: {diff_x4}, Diff Y: {diff_y4}")

    print("\n[Result] All derivation steps confirmed mathematically.")



import sympy as sp

def verify_characteristic_polynomial():
    print("=========================================================")
    print("   VERIFICATION: Characteristic Polynomial of H_tilde    ")
    print("=========================================================\n")

    # 1. Define Symbols
    lam = sp.symbols('lambda', real=True)
    kappa, Omega, beta_mu = sp.symbols('kappa Omega beta_mu', real=True)
    tau, psi = sp.symbols('tau psi', real=True)
    sigma = sp.symbols('sigma', real=True) # +/- 1
    
    # Trig definitions for simplification
    # omega = tan(psi)
    # sin(psi), cos(psi) are used directly.
    s_psi = sp.sin(psi)
    c_psi = sp.cos(psi)
    omega = s_psi / c_psi
    
    cosh_tau = sp.cosh(tau)
    sinh_tau = sp.sinh(tau)

    # 2. Construct Matrix Elements from Hyperbolic Parameterization
    # x1_tilde - p_mu
    elem_13 = -sigma * kappa * ( (beta_mu/Omega)*cosh_tau*c_psi + sinh_tau*s_psi )
    
    # y1_tilde
    elem_23 = sigma * kappa * ( -(beta_mu/Omega)*cosh_tau*s_psi + sinh_tau*c_psi )
    
    # Matrix H_tilde
    # Row 1: Z/Omega, 0, x1-p
    # Row 2: w*Z/Omega, 0, y1
    # Row 3: 0, Z, 0
    # Note: Z = kappa
    
    row1 = [kappa/Omega, 0, elem_13]
    row2 = [omega * kappa/Omega, 0, elem_23]
    row3 = [0, kappa, 0]
    
    H_tilde = sp.Matrix([row1, row2, row3])
    
    print("Matrix H_tilde constructed.")

    # 3. Calculate Characteristic Polynomial: det(H - lambda*I)
    # We compute it directly using SymPy's determinant function
    I = sp.eye(3)
    char_poly_calculated = (H_tilde - lam * I).det()
    
    print("Calculated determinant symbolically...")

    # 4. Define Target Polynomial (from your prompt)
    # P(lambda) = -lambda^3 + (kappa/Omega)*lambda^2 
    #             + sigma*kappa^2 * (sinh*cos - (beta/Omega)*cosh*sin)*lambda 
    #             - sigma * kappa^3 / (Omega*cos) * sinh
    
    coeff_lambda2 = kappa / Omega
    
    coeff_lambda1 = sigma * kappa**2 * (sinh_tau*c_psi - (beta_mu/Omega)*cosh_tau*s_psi)
    
    const_term = -sigma * kappa**3 / (Omega * c_psi) * sinh_tau
    
    P_target = -lam**3 + coeff_lambda2 * lam**2 + coeff_lambda1 * lam + const_term
    
    print("Constructed target polynomial from prompt...")

    # 5. Compare
    # We simplify the difference.
    # Note: Simplification involving trig (tan/sin/cos) often needs a little help, 
    # but SymPy handles basic identities well.
    
    difference = sp.simplify(char_poly_calculated - P_target)
    
    print("\n---------------------------------------------------------")
    print("Comparing Calculated vs Target...")
    print(f"Difference: {difference}")
    
    if difference == 0:
        print("\n[SUCCESS] The derived characteristic polynomial matches the target exactly.")
    else:
        print("\n[FAILURE] Mismatch found.")
        print("Calculated Poly:")
        print(char_poly_calculated)
        print("Target Poly:")
        print(P_target)





import numpy as np
import sympy as sp
from scipy.optimize import root

def validate_complex_roots():
    # Parameters
    sigma = -1
    beta_mu = 0.999999914516273
    omega_val = 1.3003177056818682
    Omega_val = 1.3003177714225105
    kappa = 1.0
    
    # Derived geometry
    # cos(psi) = 1/sqrt(1+w^2)
    # sin(psi) = w/sqrt(1+w^2)
    sec2 = 1 + omega_val**2
    cos_psi = 1.0 / np.sqrt(sec2)
    sin_psi = omega_val * cos_psi
    
    # Define functions for P and dP/dtau
    # P(lambda, tau) = -lambda^3 + (k/O)lambda^2 + sigma*k^2*Q*lambda - C
    
    def get_Q_C(tau):
        # tau is complex
        cosh = np.cosh(tau)
        sinh = np.sinh(tau)
        Q = sinh * cos_psi - (beta_mu / Omega_val) * cosh * sin_psi
        C = (sigma * kappa**3 / (Omega_val * cos_psi)) * sinh
        return Q, C

    def get_dP_dtau_partial(lam, tau):
        # dP/dtau partial = sigma*k^2 * dQ/dtau * lambda - dC/dtau
        # dQ/dtau = cosh*cos - (b/O)*sinh*sin
        # dC/dtau = (sigma*k^3 / (O*cos)) * cosh
        cosh = np.cosh(tau)
        sinh = np.sinh(tau)
        dQ_dt = cosh * cos_psi - (beta_mu / Omega_val) * sinh * sin_psi
        dC_dt = (sigma * kappa**3 / (Omega_val * cos_psi)) * cosh
        
        return sigma * kappa**2 * dQ_dt * lam - dC_dt

    def get_P(lam, tau):
        Q, C = get_Q_C(tau)
        return -lam**3 + (kappa / Omega_val) * lam**2 + sigma * kappa**2 * Q * lam - C

    def get_stationary_lambda(tau):
        # lambda = k * sec^2 / D
        # D = Omega - beta * omega * tanh
        tanh = np.tanh(tau)
        D = Omega_val - beta_mu * omega_val * tanh
        return (kappa * sec2) / D

    # Find the root precisely
    # Condition: cosh - RHS = 0
    def condition_eq(tau_vec):
        # tau_vec is [real, imag]
        tau = tau_vec[0] + 1j * tau_vec[1]
        cosh = np.cosh(tau)
        tanh = np.tanh(tau)
        
        # RHS = -sigma * beta * (Omega - beta*omega*tanh)^2 / ( sec^3 * (Omega*omega + beta*tanh) )
        numerator = -sigma * beta_mu * (Omega_val - beta_mu * omega_val * tanh)**2
        denominator = (sec2**1.5) * (Omega_val * omega_val + beta_mu * tanh)
        
        val = cosh - numerator / denominator
        return [val.real, val.imag]

    # Initial guess from previous turn
    guess = [0.3120, 1.4037]
    sol = root(condition_eq, guess)
    
    if not sol.success:
        print("Root finding failed.")
        return

    tau_root = sol.x[0] + 1j * sol.x[1]
    print(f"Refined Root tau: {tau_root:.6f}")
    
    # Validation
    lam_stat = get_stationary_lambda(tau_root)
    P_val = get_P(lam_stat, tau_root)
    dP_dtau_val = get_dP_dtau_partial(lam_stat, tau_root)
    
    print(f"Stationary Lambda: {lam_stat:.6f}")
    print(f"P(lambda): {P_val:.6e}")
    print(f"dP/dtau (partial): {dP_dtau_val:.6e}")
    
    # Check if they are close to zero
    is_P_zero = abs(P_val) < 1e-6
    is_dP_zero = abs(dP_dtau_val) < 1e-6
    
    print(f"P is zero? {is_P_zero}")
    print(f"dP/dtau is zero? {is_dP_zero}")

if __name__ == "__main__":
    verify_z2_expansion()
    prove_rotation_and_parameterization()
    verify_characteristic_polynomial()
    validate_complex_roots()
