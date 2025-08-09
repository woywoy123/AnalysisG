from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import numpy as np
import os

def read_matrix(filename):
    return np.genfromtxt(filename + ".csv", delimiter=',')

def add_ellipse(ax, pts, l, lw, lbl):
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], l, linewidth = lw, label = lbl) 

def add_center(ax, elp, s, c, m, l):
    ax.scatter(elp.C[0], elp.C[1], elp.C[2], s = s, c = c, marker = m, label = l)

def add_normals(ax, elp, nrm, c, l):
    ax.quiver(elp.C[0], elp.C[1], elp.C[2], nrm[0], nrm[1], nrm[2], length=0.01, color=c, linewidth=1.5, label=l)

def add_vector(ax, elp, pts, c, m, tl):
    ax.quiver(
            elp.C[0], elp.C[1], elp.C[2], 
            pts[0]-elp.C[0], pts[1]-elp.C[1], pts[2]-elp.C[2], 
            color=c, linewidth=1.5, arrow_length_ratio=0.1
    )
    ax.scatter(pts[0], pts[1], pts[2], s=100, c=c, marker=m)
    ax.text(   pts[0], pts[1], pts[2], tl, color='black', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

def add_line(ax, pts):
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'g-', linewidth=2, label='Intersection Line')

def add_distance(ax, pts1, pts2, idx, mrk, lbl):
    ax.plot(
            [pts1[idx][0], pts2[idx][0]], 
            [pts1[idx][1], pts2[idx][1]], 
            [pts1[idx][2], pts2[idx][2]], 
            mrk, linewidth=2, label=lbl
    )
 

class Ellipse:
    def __init__(self, index):
        self.H = read_matrix("data/H", str(index))


class KinematicEllipse:
    def __init__(self, index):
        self.H = read_matrix("data/H" + str(index))
        self.C = self.H @ np.array([0, 0, 1])
        self.A = self.H @ np.array([1, 0, 0])
        self.B = self.H @ np.array([0, 1, 0])
        self.normal = np.cross(self.A, self.B)
        
        # Magnitudes of axes
        self.a = np.linalg.norm(self.A)**2  # Semi-major axis
        self.b = np.linalg.norm(self.B)**2  # Semi-minor axis
    
    def point(self, phi):     return self.H @ np.array([np.cos(phi), np.sin(phi), 1])
    def plane_equation(self): return self.normal, np.dot(self.normal, self.C)
    def generate_ellipse(self, n_points=100):
        phis = np.linspace(0, 2*np.pi, n_points)
        return np.array([self.point(phi) for phi in phis])

    def solve_intersection(self, r0, d):
        # Projection coefficients
        beta  = np.dot(d, self.A) / self.a
        delta = np.dot(d, self.B) / self.b
        alpha = np.dot(r0 - self.C, self.A) / self.a
        gamma = np.dot(r0 - self.C, self.B) / self.b
        
        # Quadratic equation coefficients
        A_coeff = beta**2 + delta**2
        B_coeff = 2 * (alpha*beta + gamma*delta)
        C_coeff = alpha**2 + gamma**2 - 1
        
        discriminant = abs(B_coeff**2 - 4*A_coeff*C_coeff)
        if discriminant < 0: return [], [], []
        s1 = (-B_coeff + np.sqrt(discriminant)) / (2*A_coeff)
        s2 = (-B_coeff - np.sqrt(discriminant)) / (2*A_coeff)
        
        phi_vals = []
        points = []
        for s in [s1, s2]:
            x_val = alpha + beta*s
            y_val = gamma + delta*s
            phi = np.arctan2(y_val, x_val) % (2*np.pi)
            point = r0 + s*d
            phi_vals.append(phi)
            points.append(point)
        return phi_vals, points, [s1, s2]

def compute_line_intersection(el1, el2): 
    d = np.cross(el1.normal, el2.normal)
    
    N1_dot_N1 = np.dot(el1.normal, el1.normal)
    N2_dot_N2 = np.dot(el2.normal, el2.normal)
    N1_dot_N2 = np.dot(el1.normal, el2.normal)
    a1 = np.dot(el1.normal, el1.C)
    a2 = np.dot(el2.normal, el2.C)
    
    r0  = ((a1 * N2_dot_N2 - a2 * N1_dot_N2) * el1.normal + (a2 * N1_dot_N1 - a1 * N1_dot_N2) * el2.normal)
    r0 /= (N1_dot_N1 * N2_dot_N2 - N1_dot_N2**2)
    return r0, d

def plot_plane(ax, point, normal, color, size=100, alpha=0.2):
    normal = normal / np.linalg.norm(normal)
    if abs(normal[0]) > 1e-6 or abs(normal[1]) > 1e-6: l = [-normal[1], normal[0], 0]
    else: l = [1, 0, 0]
    perp1 = np.array(l)
    perp1 = perp1 / np.linalg.norm(perp1)

    perp2 = np.cross(normal, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    
    # Generate grid
    u, v = np.meshgrid(np.linspace(-size, size, 10), np.linspace(-size, size, 10))
    x = point[0] + perp1[0]*u + perp2[0]*v
    y = point[1] + perp1[1]*u + perp2[1]*v
    z = point[2] + perp1[2]*u + perp2[2]*v
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

def compute_distance_metrics(points1, points2, s_values1, s_values2):
    if len(points1) < 2 or len(points2) < 2: return None, None, None
    d_plus_sq = np.linalg.norm(points1[0] - points2[0])**2
    d_minus_sq = np.linalg.norm(points1[1] - points2[1])**2
    
    mean_sq = 0.5 * (d_plus_sq + d_minus_sq)
    asymmetry = abs(d_plus_sq - d_minus_sq) / mean_sq if mean_sq > 1e-6 else 0
    return d_plus_sq, d_minus_sq, asymmetry

if __name__ == "__main__":
    # Use your explicit parameters
    ellipse1 = KinematicEllipse(1)
    ellipse2 = KinematicEllipse(2)
    ellipse3 = KinematicEllipse(3)
    
    n1, d1 = ellipse1.plane_equation()
    n2, d2 = ellipse2.plane_equation()
    n3, d3 = ellipse3.plane_equation()
   
    r01, d01_vec = compute_line_intersection(ellipse1, ellipse2)
    r02, d02_vec = compute_line_intersection(ellipse1, ellipse3)
    r12, d12_vec = compute_line_intersection(ellipse2, ellipse3)

    # Find line-ellipse intersections
    phi01_vals, points01, s_values01 = ellipse1.solve_intersection(r01, d01_vec)
    phi02_vals, points02, s_values02 = ellipse2.solve_intersection(r02, d02_vec)
    phi12_vals, points12, s_values12 = ellipse3.solve_intersection(r12, d12_vec)
   
    # Generate points on the line
    s_vals = np.linspace(-0.0001, 0.0001, 100)
    line01_pts = np.array([r01 + s*d01_vec for s in s_vals])
    line02_pts = np.array([r02 + s*d02_vec for s in s_vals])
    line12_pts = np.array([r12 + s*d12_vec for s in s_vals])

    # Generate ellipse points
    ellipse1_pts = ellipse1.generate_ellipse()
    ellipse2_pts = ellipse2.generate_ellipse()
    ellipse3_pts = ellipse3.generate_ellipse()

    all_pts = np.vstack((ellipse1_pts, ellipse2_pts, ellipse3_pts))
    xmax, xmin = max(all_pts[:,0]), min(all_pts[:,0])
    ymax, ymin = max(all_pts[:,1]), min(all_pts[:,1])
    zmax, zmin = max(all_pts[:,2]), min(all_pts[:,2])

    d_plus_sq, d_minus_sq, asymmetry = compute_distance_metrics(points01, points02, s_values01, s_values02)
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ellipses
    add_ellipse(ax, ellipse1_pts, "b-", 2, f'Ellipse 1')
    add_ellipse(ax, ellipse2_pts, "r-", 2, f'Ellipse 2')
    add_ellipse(ax, ellipse3_pts, "y-", 2, f'Ellipse 3')
   
    # Plot centers
    add_center(ax, ellipse1, 80, 'blue'  , 'o', 'Center 1')
    add_center(ax, ellipse2, 80, 'red'   , 'o', 'Center 2')
    add_center(ax, ellipse3, 80, 'yellow', 'o', 'Center 3')
   
    # Plot planes
    plot_plane(ax, ellipse1.C, n1, 'blue')
    plot_plane(ax, ellipse2.C, n2, 'red')
    plot_plane(ax, ellipse3.C, n3, 'yellow')
    
    # Plot normals
    add_normals(ax, ellipse1, n1, 'darkblue', 'Normal 1')
    add_normals(ax, ellipse2, n2, 'darkred' , 'Normal 2')
    add_normals(ax, ellipse3, n3, 'darkgreen' , 'Normal 3')
    add_line(ax, line01_pts)
    add_line(ax, line02_pts)
    add_line(ax, line12_pts)

    colors = ['cyan', 'magenta', "purple"]
    markers = ['o', 's', 'o']
    
    for i, point in enumerate(points01): add_vector(ax, ellipse1, point, colors[i], markers[i], f'φ1_{i+1}={phi01_vals[i]:.2f}\ns={s_values01[i]:.2f}')
    for i, point in enumerate(points02): add_vector(ax, ellipse2, point, colors[i], markers[i], f'φ2_{i+1}={phi02_vals[i]:.2f}\ns={s_values02[i]:.2f}')
    for i, point in enumerate(points12): add_vector(ax, ellipse3, point, colors[i], markers[i], f'φ2_{i+1}={phi12_vals[i]:.2f}\ns={s_values12[i]:.2f}')
   
    # Plot distance lines if we have two points per ellipse
#    if len(points1) == 2 and len(points2) == 2:
#        add_distance(ax, points1, points2, 0, "m--",'d²⁺ = {d_plus_sq:.4f}') 
#        add_distance(ax, points1, points2, 1, "c--",'d²⁺ = {d_minus_sq:.4f}') 
    
    # Configure plot
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title('3D Ellipse Intersection Analysis with Distance Metrics', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("\nEllipse 1 Intersections:")
    for i, (phi, point, s) in enumerate(zip(phi01_vals, points01, s_values01)):
        print(f"  Point {i+1}: φ = {phi:.4f} rad, s = {s:.4f}, Position: ({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f})")
    
    print("\nEllipse 2 Intersections:")
    for i, (phi, point, s) in enumerate(zip(phi02_vals, points02, s_values02)):
        print(f"  Point {i+1}: φ = {phi:.4f} rad, s = {s:.4f}, Position: ({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f})")

    print("\nEllipse 3 Intersections:")
    for i, (phi, point, s) in enumerate(zip(phi12_vals, points12, s_values12)):
        print(f"  Point {i+1}: φ = {phi:.4f} rad, s = {s:.4f}, Position: ({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f})")
 

    if d_plus_sq is not None:
        print("\nDistance Metrics:")
        print(f"  d²⁺ = {d_plus_sq:.6f} (squared distance between first solutions)")
        print(f"  d²⁻ = {d_minus_sq:.6f} (squared distance between second solutions)")
        print(f"  Asymmetry Ratio = {asymmetry:.6f}")
        
        # Compute the actual distances
        d_plus = np.sqrt(d_plus_sq)
        d_minus = np.sqrt(d_minus_sq)
        print(f"\nActual Distances:")
        print(f"  d⁺ = {d_plus:.6f}")
        print(f"  d⁻ = {d_minus:.6f}")
        print(f"  Difference: {abs(d_plus - d_minus):.6f}")
        
