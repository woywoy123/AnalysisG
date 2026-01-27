#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.optimize import fsolve
#
## --- 1. System Parameters (From Data Point 1) ---
#u_target = -0.364646534771
#beta_mu = 0.999999989221
#tan_psi = 0.986682765545
#Omega = 0.986682776469
#
## Derived
#psi = np.arctan(tan_psi)
#primary_invariant = (Omega**2 + beta_mu**2)**3
#
## --- 2. Basis Functions (Linear Trajectory) ---
#def get_alphas(u):
#    ap = Omega * tan_psi + beta_mu * u
#    am = Omega - beta_mu * u * tan_psi
#    return ap, am
#
## --- 3. Characteristic Equations ---
#def primary_surface_val(ap, am):
#    """Equation for the Ellipse: LHS - Invariant"""
#    if ap == 0 or am == 0: return np.nan
#    tan_phi = ap / am
#    # tan(phi - psi)
#    tan_diff = (tan_phi - tan_psi) / (1 + tan_phi * tan_psi)
#    lhs = (am**4 / ap**2) * (beta_mu**2 - Omega**2 * tan_diff**2)
#    return lhs
#
## Get the coordinates at the root to identify the specific Conjugated Hyperbola
#root_ap, root_am = get_alphas(u_target)
#
#def conjugated_surface_val(ap, am, target_val=None):
#    """Equation for the Conjugated Hyperbola"""
#    if ap == 0 or am == 0: return np.nan
#    tan_phi = ap / am
#    # tan(phi + psi) - Note the PLUS sign for conjugated
#    tan_sum = (tan_phi + tan_psi) / (1 - tan_phi * tan_psi)
#    
#    # The form is alpha_+^4 / alpha_-^2 * coupling
#    lhs = (ap**4 / am**2) * (beta_mu**2 - Omega**2 * tan_sum**2)
#    
#    if target_val is not None:
#        return lhs - target_val
#    return lhs
#
## Calculate the Invariant for the Conjugated Surface at the root
#conjugated_invariant_at_root = conjugated_surface_val(root_ap, root_am)
#
## --- 4. u-Space Master Equation ---
#def master_equation_u(u):
#    ap, am = get_alphas(u)
#    if abs(ap) < 1e-5: return np.nan
#    lhs = (am**4 / ap**2) * beta_mu**2 * (1 - u**2)
#    return lhs - primary_invariant
#
## --- 5. Plotting ---
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
#
## A. Phase Space Plot
#x_range = np.linspace(0.01, 3.0, 400)
#y_range = np.linspace(0.01, 3.0, 400)
#X, Y = np.meshgrid(x_range, y_range)
#
## Evaluate surfaces on grid
#Z_primary = np.array([[primary_surface_val(x, y) - primary_invariant for x in x_range] for y in y_range])
#Z_conjugated = np.array([[conjugated_surface_val(x, y, conjugated_invariant_at_root) for x in x_range] for y in y_range])
#
## Contours
#ax1.contour(X, Y, Z_primary, levels=[0], colors='blue', linewidths=2, label='Primary (Ellipse)')
#ax1.contour(X, Y, Z_conjugated, levels=[0], colors='green', linewidths=2, linestyles='--', label='Conjugated (Hyperbola)')
#
## Trajectory
#u_vals = np.linspace(-1.0, 0.5, 100)
#traj_ap, traj_am = get_alphas(u_vals)
#ax1.plot(traj_ap, traj_am, 'r-', linewidth=1.5, label='Linear Trajectory')
#
## Root
#ax1.plot(root_ap, root_am, 'ko', markersize=10, zorder=10, label='ROOT (Intersection)')
#
#ax1.set_xlabel(r'$\alpha_+$')
#ax1.set_ylabel(r'$\alpha_-$')
#ax1.set_title('1. Phase Space: The Intersection of Trajectory, Ellipse, and Hyperbola')
#ax1.grid(True)
## Create custom legend handles since contour doesn't support label kwarg directly in all versions
#from matplotlib.lines import Line2D
#custom_lines = [Line2D([0], [0], color='blue', lw=2),
#                Line2D([0], [0], color='green', lw=2, linestyle='--'),
#                Line2D([0], [0], color='red', lw=2),
#                Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=10)]
#ax1.legend(custom_lines, ['Real Ellipse', 'Conjugated Hyperbola', 'Trajectory', 'Root'])
#
## B. u-Space Roots Plot
#u_space = np.linspace(-0.9, 0.9, 500)
#lhs_vals = [(get_alphas(u)[1]**4 / get_alphas(u)[0]**2) * beta_mu**2 * (1 - u**2) for u in u_space]
#
#ax2.plot(u_space, lhs_vals, 'b-', label='LHS (Geometric Factor)')
#ax2.axhline(y=primary_invariant, color='r', linestyle='--', label='RHS (Invariant Mass)')
#ax2.plot(u_target, primary_invariant, 'ko', markersize=8, label='Root')
#
#ax2.set_xlabel('u')
#ax2.set_ylabel('Magnitude')
#ax2.set_title('2. Roots in u-Space')
#ax2.set_ylim(0, primary_invariant * 2)
#ax2.grid(True)
#ax2.legend()
#
#plt.tight_layout()
#plt.show()
#
#print(f"--- Proof of Intersection ---")
#print(f"Root u: {u_target}")
#print(f"Coordinates at Root: alpha_+ = {root_ap:.4f}, alpha_- = {root_am:.4f}")
#print(f"Primary Eq Error:    {primary_surface_val(root_ap, root_am) - primary_invariant:.4e}")
#print(f"Conjugated Eq Val:   {conjugated_invariant_at_root:.4e}")


#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.optimize import fsolve
#
## --- 1. System Parameters (From User Data Point 1) ---
#u_target = -0.364646534771
#beta_mu = 0.999999989221
#tan_psi = 0.986682765545
#Omega = 0.986682776469
#
## Derived Parameters
#psi = np.arctan(tan_psi)
#invariant_sum = (Omega**2 + beta_mu**2)**3
#invariant_diff = (Omega**2 - beta_mu**2)**3
#
## --- 2. Define Basis Functions (Linear Trajectory) ---
#def get_alphas(u):
#    """Returns alpha_plus, alpha_minus based on linear trajectory parameter u."""
#    a_plus = Omega * tan_psi + beta_mu * u
#    a_minus = Omega - beta_mu * u * tan_psi
#    return a_plus, a_minus
#
## --- 3. Define Characteristic Surfaces (Implicit Equations) ---
#def primary_surface_error(ap, am):
#    """
#    Returns 0 if point (ap, am) is on the Primary Surface.
#    Equation: (am^4 / ap^2) * (beta_mu^2 - Omega^2 * tan^2(phi - psi)) = (Omega^2 + beta_mu^2)^3
#    """
#    if ap == 0 or am == 0: return np.nan
#    tan_phi = ap / am
#    # tan(phi - psi) identity
#    tan_diff = (tan_phi - tan_psi) / (1 + tan_phi * tan_psi)
#    
#    lhs = (am**4 / ap**2) * (beta_mu**2 - Omega**2 * tan_diff**2)
#    return lhs - invariant_sum
#
#def conjugated_surface_error(ap, am):
#    """
#    Returns 0 if point (ap, am) is on the Conjugated Surface.
#    Equation: (ap^4 / am^2) * (beta_mu^2 - Omega^2 * tan^2(phi + psi)) = (Omega^2 - beta_mu^2)^3
#    """
#    if ap == 0 or am == 0: return np.nan
#    tan_phi = ap / am
#    # tan(phi + psi) identity
#    tan_sum = (tan_phi + tan_psi) / (1 - tan_phi * tan_psi)
#    
#    lhs = (ap**4 / am**2) * (beta_mu**2 - Omega**2 * tan_sum**2)
#    return lhs - invariant_diff
#
## --- 4. Define u-Space Master Equation ---
#def master_equation_u(u):
#    """
#    The scalar polynomial derived analytically.
#    LHS(u) = (alpha_-^4 / alpha_+^2) * beta_mu^2 * (1 - u^2)
#    Target = (Omega^2 + beta_mu^2)^3
#    """
#    ap, am = get_alphas(u)
#    # Avoid division by zero in plot sweep
#    if abs(ap) < 1e-5: return np.nan
#    
#    lhs = (am**4 / ap**2) * beta_mu**2 * (1 - u**2)
#    return lhs - invariant_sum
#
## --- 5. Generate Plots ---
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
#
## Plot 1: Phase Space (Alpha Plane)
## Create grid
#x_range = np.linspace(0.00000000000000000001, 3.0, 400)
#y_range = np.linspace(0.00000000000000000001, 3.0, 400)
#X, Y = np.meshgrid(x_range, y_range)
#
## Calculate surface errors on grid
#Z1 = np.array([[primary_surface_error(x, y) for x in x_range] for y in y_range])
#Z2 = np.array([[conjugated_surface_error(x, y) for x in x_range] for y in y_range])
#
## Contour Plot
#ax1.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)#, label='Primary (Ellipse)')
#ax1.contour(X, Y, Z2, levels=[0], colors='green', linewidths=2, linestyles='--')#, label='Conjugated')
#
## Plot Trajectory
#u_vals = np.linspace(-1.5, 1.5, 100)
#traj_ap, traj_am = get_alphas(u_vals)
#ax1.plot(traj_ap, traj_am, 'r-', label='Linear Trajectory α(u)')
#
## Mark the specific root
#root_ap, root_am = get_alphas(u_target)
#ax1.plot(root_ap, root_am, 'ko', markersize=8, label=f'Root u={u_target:.4f}')
#
#ax1.set_xlabel(r'$\alpha_+$')
#ax1.set_ylabel(r'$\alpha_-$')
#ax1.set_title('1. Phase Space: Intersection of Trajectory and Surfaces')
#ax1.grid(True)
#ax1.legend()
#
## Plot 2: Hyperbolic System relative to u
#u_plot = np.linspace(-0.9, 0.9, 500)
#eq_vals = [master_equation_u(u) + invariant_sum for u in u_plot] # Add invariant back to show LHS vs RHS
#
#ax2.plot(u_plot, eq_vals, 'b-', label='LHS (Geometric Factor)')
#ax2.axhline(y=invariant_sum, color='r', linestyle='--', label='RHS (Invariant Mass)')
#ax2.plot(u_target, invariant_sum, 'ko', label='Intersection')
#
#ax2.set_xlabel('u (Rapidity/Boost)')
#ax2.set_ylabel('Magnitude')
#ax2.set_title('2. Roots in u-Space (LHS vs Invariant)')
#ax2.set_ylim(0, invariant_sum * 2)
#ax2.grid(True)
#ax2.legend()
#
#plt.tight_layout()
#plt.show()
#
## --- 6. Numerical Verification ---
## Solve for root near the target
#calculated_root = fsolve(master_equation_u, -0.3)[0]
#
#print(f"--- Verification Results ---")
#print(f"Target u (Data):      {u_target:.12f}")
#print(f"Calculated u (Eq):    {calculated_root:.12f}")
#print(f"Difference:           {abs(u_target - calculated_root):.12e}")
#print(f"Is Point on Surface?: {abs(master_equation_u(calculated_root)) < 1e-6}")

#exit()

from AnalysisG import Analysis
from AnalysisG.events.bsm_4tops import BSM4Tops
from AnalysisG.selections.neutrino.conuix.conuix import Conuix

smpl = "../../../test/samples/dilepton/DAOD_TOPQ1.21955717._000001.root"

x = BSM4Tops()
s = Conuix()

ana = Analysis()
ana.AddSamples(smpl,"tmp")
ana.AddEvent(x, "tmp")
ana.AddSelection(s)
ana.Threads = 1
ana.DebugMode = True
ana.Start()


