# plot_mobius_sextic.py
# Requires: numpy, sympy, matplotlib
# Example: python plot_mobius_sextic.py  (or paste into a Jupyter cell)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
import sys

# ---------- PARAMETERS (edit these as desired) ----------
psi = 0.6         # radians (choose any real)
beta_mu = 0.8     # real
# -------------------------------------------------------

omega = np.tan(psi)
Omega_sq = 1.0 + omega**2 - beta_mu**2
if Omega_sq <= 0:
    raise SystemExit("Omega^2 <= 0 for these parameters; choose psi and beta_mu so Omega^2 > 0.")
Omega = np.sqrt(Omega_sq)

# ---------- Build symbolic sextic P(M) ----------
M = sp.symbols('M', real=True)
cpsi = sp.cos(psi)
spsi = sp.sin(psi)

P_expr = M**2*(cpsi + M*spsi)**4 - Omega**2 * cpsi**2 * (cpsi + M*spsi)**2 + Omega**4 * cpsi**4 * (1 + M**2)
P_poly = sp.expand(P_expr)
coeffs = sp.Poly(P_poly, M).all_coeffs()
coeffs_num = [float(c) for c in coeffs]   # highest-first

# ---------- Find numeric roots ----------
roots = np.roots(coeffs_num)

# ---------- Evaluate admissibility for real roots ----------
admissible = []
all_real_roots = []
tol = 1e-9
for r in roots:
    if abs(r.imag) < 1e-9:
        Mr = float(r.real)
        all_real_roots.append(Mr)
        denom_for_u = cpsi + Mr * spsi
        if abs(denom_for_u) < 1e-12:
            # singular: u infinite, skip
            admissible.append((Mr, None, None, None, False, "denom_u_zero"))
            continue
        # u from formula (9)
        u_r = (Omega / beta_mu) * (Mr * cpsi - spsi) / denom_for_u
        in_domain = abs(u_r) < 1 - 1e-12
        # denominator in Möbius fraction (2)
        D = Omega*cpsi - beta_mu * u_r * spsi
        denom_ok = abs(D) > 1e-12
        # compute residual of original equation (1) if possible
        if in_domain and denom_ok:
            LHS = beta_mu * cpsi**2 * np.sqrt(max(0.0, 1 - u_r**2)) + (Omega*spsi + beta_mu * u_r * cpsi) / (D**2)
            # sign check for unsquared relation M = -beta cos^2 psi sqrt(1-u^2) * D
            rhs = -beta_mu * cpsi**2 * np.sqrt(max(0.0, 1 - u_r**2)) * D
            sign_ok = abs(Mr - rhs) < 1e-6
        else:
            LHS = np.nan
            sign_ok = False
        admissible.append((Mr, u_r, np.arctanh(u_r) if in_domain else None, LHS, sign_ok, "ok" if denom_ok and in_domain else "domain_fail"))

# ---------- Prepare Möbius strip surface ----------
theta_vals = np.linspace(0, 2*np.pi, 120)
v_vals = np.linspace(-1, 1, 60)
Theta, V = np.meshgrid(theta_vals, v_vals)

X = (1 + 0.5 * V * np.cos(Theta/2)) * np.cos(Theta)
Y = (1 + 0.5 * V * np.cos(Theta/2)) * np.sin(Theta)
Z = 0.5 * V * np.sin(Theta/2)

# ---------- Graph of Möbius map M(tau) (the full graph, usually NOT the solution set) ----------
tau_curve = np.linspace(-3.0, 3.0, 400)
tanh_tau = np.tanh(tau_curve)
M_of_tau = (Omega*spsi + beta_mu * np.cos(psi) * tanh_tau) / (Omega*cpsi - beta_mu * spsi * tanh_tau)
theta_of_tau = np.arctan(M_of_tau)
V_of_tau = tanh_tau
Xc = (1 + 0.5 * V_of_tau * np.cos(theta_of_tau/2)) * np.cos(theta_of_tau)
Yc = (1 + 0.5 * V_of_tau * np.cos(theta_of_tau/2)) * np.sin(theta_of_tau)
Zc = 0.5 * V_of_tau * np.sin(theta_of_tau/2)

# ---------- Points corresponding to admissible roots (mapped onto strip) ----------
points = []
for Mr, u_r, tau_r, LHS, sign_ok, status in admissible:
    if u_r is None:
        continue
    # map M->theta and u->v
    theta_r = np.arctan(Mr)
    v_r = u_r
    Xr = (1 + 0.5 * v_r * np.cos(theta_r/2)) * np.cos(theta_r)
    Yr = (1 + 0.5 * v_r * np.cos(theta_r/2)) * np.sin(theta_r)
    Zr = 0.5 * v_r * np.sin(theta_r/2)
    points.append((Mr, u_r, tau_r, LHS, sign_ok, status, Xr, Yr, Zr))

# ---------- Plot ----------
fig = plt.figure(figsize=(11,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f"Möbius strip & sextic roots: psi={psi:.3f}, beta_mu={beta_mu:.3f}, Omega={Omega:.3f}")

ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0, antialiased=True, alpha=0.9)
ax.plot(Xc, Yc, Zc, label='graph of M(tau)')

# plot numeric root points: 'o' if sign_ok True, 'x' otherwise
for (Mr, u_r, tau_r, LHS, sign_ok, status, Xr, Yr, Zr) in points:
    if sign_ok:
        ax.scatter([Xr], [Yr], [Zr], s=80, marker='o')
    else:
        ax.scatter([Xr], [Yr], [Zr], s=80, marker='x')

# text summary in lower-left
text_lines = [f"psi={psi:.6f}, beta_mu={beta_mu:.6f}, Omega={Omega:.6f}", "", "Real roots and checks:"]
for (Mr, u_r, tau_r, LHS, sign_ok, status) in admissible:
    text_lines.append(f"M={Mr: .6f}, u={'nan' if u_r is None else f'{u_r:.6f}'}, tau={'nan' if tau_r is None else f'{tau_r:.6f}'}, LHS_resid={'nan' if LHS is None else f'{LHS:.3e}'}, sign_ok={sign_ok}, status={status}")
text = "\n".join(text_lines)
ax.text2D(0.02, 0.02, text, transform=ax.transAxes, fontsize=9)

ax.view_init(elev=25, azim=-60)
plt.tight_layout()
plt.show()

# ---------- Print summary to console ----------
print("Parameters: psi =", psi, "beta_mu =", beta_mu, "Omega =", Omega)
print("\nPolynomial coefficients (highest-first):")
print(coeffs_num)
print("\nNumeric roots (all):")
print(roots)
print("\nAdmissibility checks (M, u, tau, LHS_resid, sign_ok, status):")
for item in admissible:
    print(item)

