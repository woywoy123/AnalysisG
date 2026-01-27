import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import math


def projections(points, ax1, ax2, ax3, color='blue', style = "-", linewidth=2):
    ax1.plot(points[:, 0], points[:, 1], color=color, linestyle = style, linewidth=linewidth)
    ax1.set_xlabel('x (GeV)')
    ax1.set_ylabel('y (GeV)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # XZ projection
    ax2.plot(points[:, 0], points[:, 2], color=color, linestyle = style, linewidth=linewidth)
    ax2.set_xlabel('x (GeV)')
    ax2.set_ylabel('z (GeV)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # YZ projection
    ax3.plot(points[:, 1], points[:, 2], color=color, linestyle = style, linewidth=linewidth)
    ax3.set_xlabel('y (GeV)')
    ax3.set_ylabel('z (GeV)')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')


def compute_ellipse_points(H, num_points=100):
    t_values = np.linspace(0, 2*np.pi, num_points)
    points = []
    
    for t in t_values:
        t_vec = np.array([np.cos(t), np.sin(t), 1])
        p_nu = H.dot(t_vec)
        points.append(p_nu)
    return np.array(points)


def live_ellipses(nu_sol, ax, c, l):
    projections(compute_ellipse_points(nu_sol), ax[0], ax[1], ax[2], color=c, style = l)


def plot_ellipses(nu_sol, R_T = None):
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle('Neutrino Ellipse in F Frame (H_tilde)', fontsize=14, fontweight='bold')
    for i in range(len(nu_sol)):
        H_tilde, col, ln = nu_sol[i]  # In F frame
        points_tilde = compute_ellipse_points(R_T.dot(H_tilde) if R_T is not None else H_tilde)
        projections(points_tilde, axes1[0], axes1[1], axes1[2], color=col, style = ln)
#    axes1[0].plot(nu.px, nu.py, 'r*', markersize=10, label='Truth Neutrino')
#    axes1[1].plot(nu.px, nu.pz, 'r*', markersize=10)
#    axes1[2].plot(nu.py, nu.pz, 'r*', markersize=10)
#    axes1[0].legend(loc='best')
    plt.tight_layout()
    plt.show()



