# -------------- plane-plane -> line (mutual intersection) ----------------
def mutual_intersection_line(ell1: NuSolEllipse, ell2: NuSolEllipse):
    n1, d1 = ell1.normal, float(np.dot(ell1.normal, ell1.center))
    n2, d2 = ell2.normal, float(np.dot(ell2.normal, ell2.center))
    u = np.cross(n1, n2)
    if norm(u) < 1e-12:
        return None
    u = u / (norm(u) + EPS)
    A = np.vstack([n1, n2])
    b = np.array([d1, d2], dtype=float)
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    return (p, u, (ell1.idx_pair, ell2.idx_pair))

# -------------- closest points between lines ----------------
def closest_points_between_lines(p1, d1, p2, d2):
    d1 = d1 / (norm(d1) + EPS); d2 = d2 / (norm(d2) + EPS)
    w0 = p1 - p2
    a = np.dot(d1, d1); b = np.dot(d1, d2); c = np.dot(d2, d2)
    d = np.dot(d1, w0); e = np.dot(d2, w0)
    denom = a*c - b*b
    if abs(denom) < EPS:
        # parallel-ish: pick projected points
        s = 0.0
        t = (b*d - a*e) / (b*b - a*c + EPS) if abs(b*b - a*c) > EPS else 0.0
    else:
        s = (b*e - c*d) / denom
        t = (a*e - b*d) / denom
    c1 = p1 + s*d1
    c2 = p2 + t*d2
    return c1, c2, norm(c1 - c2)

# -------------- planes from pairs of lines (no MET) ----------------
def planes_from_two_lines(line1, line2, coplanar_eps=1e-2):
    p1, d1, meta1 = line1
    p2, d2, meta2 = line2
    c1, c2, dist = closest_points_between_lines(p1, d1, p2, d2)
    planes = []
    if dist < coplanar_eps:
        n = np.cross(d1, d2)
        if norm(n) < EPS:
            n = np.cross(d1, (c2 - c1))
            if norm(n) < EPS:
                return []
        n /= (norm(n) + EPS)
        plane_point = 0.5*(c1 + c2)
        planes.append((plane_point, n))
    else:
        nA = np.cross(d1, (c2 - p1))
        if norm(nA) >= EPS:
            nA /= (norm(nA) + EPS)
            planes.append((p1, nA))
        nB = np.cross(d2, (c1 - p2))
        if norm(nB) >= EPS:
            nB /= (norm(nB) + EPS)
            planes.append((p2, nB))
    return planes

def build_planes_from_lines(lines, coplanar_eps=1e-2):
    planes = []
    m = len(lines)
    for i in range(m):
        for j in range(i+1, m):
            new = planes_from_two_lines(lines[i], lines[j], coplanar_eps=coplanar_eps)
            if new: planes.extend(new)
    return planes

# -------------- triple-plane intersections -> points ----------------
def triple_plane_intersections(planes):
    pts = []
    m = len(planes)
    for comb in combinations(range(m), 3):
        i,j,k = comb
        pA, nA = planes[i]; pB, nB = planes[j]; pC, nC = planes[k]
        N = np.vstack([nA, nB, nC])
        if abs(np.linalg.det(N)) < 1e-9: continue
        d = np.array([np.dot(nA, pA), np.dot(nB, pB), np.dot(nC, pC)], dtype=float)
        try: x = np.linalg.solve(N, d)
        except np.linalg.LinAlgError: continue
        pts.append(x)
    print(planes)
    if not pts: return np.zeros((0,3), dtype=float)
    return np.array(pts, dtype=float)

# -------------- ellipse-line quadratic intersections ----------------
def ellipse_line_intersections(ellipse: NuSolEllipse, line_origin, line_dir):
    ellipse.update_from_zt()
    c = ellipse.center; p = np.array(line_origin, dtype=float); d = np.array(line_dir, dtype=float)
    p_rel = p - c
    u = ellipse.perp1; v = ellipse.perp2
    r1, r2 = ellipse.radii[0], ellipse.radii[1]
    a_u = np.dot(d, u); a_v = np.dot(d, v)
    c_u = np.dot(p_rel, u); c_v = np.dot(p_rel, v)
    A = (a_u/r1)**2 + (a_v/r2)**2
    B = 2.0*((a_u*c_u)/(r1**2) + (a_v*c_v)/(r2**2))
    C = (c_u/r1)**2 + (c_v/r2)**2 - 1.0
    if abs(A) < EPS:
        if abs(B) < EPS:
            return []
        s = -C / B
        return [p + s*d]
    disc = B*B - 4.0*A*C
    if disc < -EPS:
        return []
    disc = max(disc, 0.0)
    sqrt_disc = math.sqrt(disc)
    s1 = (-B - sqrt_disc) / (2.0*A)
    s2 = (-B + sqrt_disc) / (2.0*A)
    pts = [p + s1*d]
    if abs(s2 - s1) > 1e-9:
        pts.append(p + s2*d)
    return pts


# Calculate the inverse covariance matrix
Sigma_inv = inv(Sigma)

# Generate random kinematic parameters for each neutrino
neutrino_params = []
for i in range(num_neutrinos):
    # Random center of kinematic ellipse
    C_K = np.random.randn(3) * 5
    
    # Random semi-major and semi-minor axes
    a = np.random.uniform(2, 5)
    b = np.random.uniform(1, 3)
    
    # Random orientation vectors A_K and B_K
    A_K = np.random.randn(3)
    A_K = A_K / np.linalg.norm(A_K) * a
    
    B_K = np.random.randn(3)
    # Ensure orthogonality with A_K
    B_K = B_K - (np.dot(B_K, A_K) / np.dot(A_K, A_K)) * A_K
    B_K = B_K / np.linalg.norm(B_K) * b
    
    # Random Möbius parameter bounds
    M_min = np.random.uniform(-2, -1)
    M_max = np.random.uniform(1, 2)
    
    neutrino_params.append({
        'C_K': C_K,
        'A_K': A_K,
        'B_K': B_K,
        'a': a,
        'b': b,
        'M_bounds': (M_min, M_max),
        'Z': np.random.uniform(0, 10),  # Initial mass squared guess
        'phi': np.random.uniform(0, 2*np.pi)  # Initial angle guess
    })

# Möbius transformation functions
def mobius_transform(u, beta_mu, omega, w):
    return (beta_mu * u + omega * w) / (omega - beta_mu * w * u)

def inverse_mobius(M, beta_mu, omega, w):
    return (omega * (M - w)) / (beta_mu * (1 + M * w))

# Function to calculate neutrino momentum from parameters
def neutrino_momentum(M, Z, phi, params, beta_mu, omega, w):
    # Calculate u from M
    u = inverse_mobius(M, beta_mu, omega, w)
    
    # Calculate hyperbolic functions
    cosh_t = 1 / np.sqrt(1 - u**2)
    sinh_t = u / np.sqrt(1 - u**2)
    
    # Calculate ellipse center
    C_K = params['C_K']
    A_K = params['A_K']
    B_K = params['B_K']
    
    # Calculate momentum
    p_nu = C_K + A_K * np.cos(phi) + B_K * np.sin(phi)
    return p_nu

# Function to find the exact intersection of a plane with an ellipsoid
def exact_plane_ellipsoid_intersection(plane_point, plane_normal, ellipsoid_center, ellipsoid_inv_cov, k=1):
    # Normalize the plane normal
    n = plane_normal / np.linalg.norm(plane_normal)
    
    # Find a point on the plane closest to the ellipsoid center
    t = np.dot(n, ellipsoid_center - plane_point)
    proj_center = ellipsoid_center - n * t
    
    # Find two orthonormal vectors in the plane
    if np.abs(n[0]) > 1e-6 or np.abs(n[1]) > 1e-6:
        v1 = np.array([-n[1], n[0], 0])
    else:
        v1 = np.array([0, -n[2], n[1]])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(n, v1)
    
    # The equation of the ellipsoid: (x - c)^T M (x - c) = k^2
    # For points in the plane: x = proj_center + u*v1 + v*v2
    # Substitute into the ellipsoid equation:
    # [u v] * [v1^T M v1, v1^T M v2; v2^T M v1, v2^T M v2] * [u; v] = k^2
    
    M = ellipsoid_inv_cov
    
    # Create the 2x2 matrix for the conic in the plane
    M11 = v1 @ M @ v1
    M12 = v1 @ M @ v2
    M22 = v2 @ M @ v2
    M_2d = np.array([[M11, M12], [M12, M22]])
    
    # The center of the ellipse in the plane coordinates is (0, 0)
    # We need to find the axes of the ellipse
    # Eigen decomposition to find the axes
    eigenvalues, eigenvectors = eig(M_2d)
    
    # The semi-axis lengths are k/sqrt(eigenvalues)
    a = k / np.sqrt(eigenvalues[0].real)
    b = k / np.sqrt(eigenvalues[1].real)
    
    # The axes directions in the plane coordinates
    axis1_2d = eigenvectors[:, 0] * a
    axis2_2d = eigenvectors[:, 1] * b
    
    # Convert back to 3D coordinates
    axis1_3d = axis1_2d[0] * v1 + axis1_2d[1] * v2
    axis2_3d = axis2_2d[0] * v1 + axis2_2d[1] * v2
    
    return proj_center, axis1_3d, axis2_3d

# Example usage with dummy parameters
beta_mu = 0.8
omega = 1.2
w = 0.5

# Run the optimization
try:
    M_opt, Z_opt, phi_opt, chi2_min = optimize_neutrino_parameters(beta_mu, omega, w)
    print("Optimization successful!")
    print(f"Minimum chi2: {chi2_min}")
    for i in range(num_neutrinos):
        print(f"Neutrino {i+1}: M = {M_opt[i]:.3f}, Z = {Z_opt[i]:.3f}, phi = {phi_opt[i]:.3f}")
except ValueError as e:
    print(f"Error: {e}")

# Visualization
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot the ellipsoid (p_miss with uncertainty)
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

ellipsoid_points = np.stack([x, y, z], axis=2)
ellipsoid_shape = ellipsoid_points.shape
ellipsoid_points = ellipsoid_points.reshape(-1, 3) @ sqrtm(Sigma)
ellipsoid_points = ellipsoid_points.reshape(ellipsoid_shape)
ellipsoid_points += p_miss

ax.plot_surface(ellipsoid_points[:, :, 0], ellipsoid_points[:, :, 1], ellipsoid_points[:, :, 2], 
                alpha=0.1, color='blue', label='Uncertainty Ellipsoid')

# Plot p_miss
ax.scatter(*p_miss, color='blue', s=100, marker='o', label='p_miss')

# Plot neutrino ellipses and their intersections with the constraint
for i, params in enumerate(neutrino_params):
    C_K = params['C_K']
    A_K = params['A_K']
    B_K = params['B_K']
    
    # Generate points on the neutrino ellipse
    neutrino_points = generate_ellipse_points(C_K, A_K, B_K)
    ax.plot(neutrino_points[:, 0], neutrino_points[:, 1], neutrino_points[:, 2], 
            label=f'Neutrino Ellipse {i+1}', linewidth=2)
    
    # Find the plane of the neutrino ellipse
    plane_normal = np.cross(A_K, B_K)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    # Find exact intersection with ellipsoid
    ellipse_center, axis1, axis2 = exact_plane_ellipsoid_intersection(
        C_K, plane_normal, p_miss, Sigma_inv)
    
    # Generate points on the constraint ellipse
    constraint_points = generate_ellipse_points(ellipse_center, axis1, axis2)
    ax.plot(constraint_points[:, 0], constraint_points[:, 1], constraint_points[:, 2], 
            '--', linewidth=2, label=f'Constraint Ellipse {i+1}')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Hierarchical Analytical-Numerical Neutrino Reconstruction (4 Neutrinos)')
ax.legend()

plt.tight_layout()
plt.show()

# Print parameters for reference
print("Parameters:")
print(f"p_miss: {p_miss}")
print(f"Covariance Matrix:\n{Sigma}")
print("\nNeutrino Parameters:")
for i, params in enumerate(neutrino_params):
    print(f"Neutrino {i+1}:")
    print(f"  Center (C_K): {params['C_K']}")
    print(f"  A_K: {params['A_K']} (length: {params['a']})")
    print(f"  B_K: {params['B_K']} (length: {params['b']})")
    print(f"  M bounds: {params['M_bounds']}")
    print()





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
   
    add_line(ax, line01_pts)
    add_line(ax, line02_pts)
    add_line(ax, line12_pts)

    colors = ['cyan', 'magenta', "purple"]
    markers = ['o', 's', 'o']
    
#    for i, point in enumerate(points01): add_vector(ax, ellipse1, point, colors[i], markers[i], f'φ1_{i+1}={phi01_vals[i]:.2f}\ns={s_values01[i]:.2f}')
#    for i, point in enumerate(points02): add_vector(ax, ellipse2, point, colors[i], markers[i], f'φ2_{i+1}={phi02_vals[i]:.2f}\ns={s_values02[i]:.2f}')
#    for i, point in enumerate(points12): add_vector(ax, ellipse3, point, colors[i], markers[i], f'φ2_{i+1}={phi12_vals[i]:.2f}\ns={s_values12[i]:.2f}')
   
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
    




class KinematicEllipse:
    def __init__(self, Z, t, w, beta_mu, R_T):
        self.Z = Z
        self.t = t
        self.w = w
        self.beta_mu = beta_mu
        self.R_T = R_T
        
        # Compute fundamental parameters
        self.Omega = np.sqrt(w**2 + 1 - beta_mu**2)
        self.sqrt_denom = np.sqrt(1 + w**2)
        
        # Compute H_bar(t) matrix components
        self.H1 = np.array([
            [1, 0, 0],
            [w, 0, 0],
            [0, self.Omega, 0]
        ])
        
        self.H2 = (beta_mu / self.sqrt_denom) * np.array([
            [0, 0, -1],
            [0, 0, -w],
            [0, 0, 0]
        ])
        
        self.H3 = (self.Omega / self.sqrt_denom) * np.array([
            [0, 0, -w],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        # Compute full H_bar(t) matrix
        self.H_bar = self.H1 + np.cosh(t) * self.H2 + np.sinh(t) * self.H3
        
        # Compute full transformation matrix
        self.H = (Z / self.Omega) * self.R_T @ self.H_bar
        self.center = self.H @ np.array([0, 0, 1])
        
        # Basis vectors (already orthogonal)
        self.A = self.H @ np.array([1, 0, 0])
        self.B = self.H @ np.array([0, 1, 0])
        self.normal = np.cross(self.A, self.B)
        
        # Magnitudes of axes
        self.a = norm(self.A)  # Semi-major axis
        self.b = norm(self.B)  # Semi-minor axis
    

def find_intersection_line(n1, d1, n2, d2, tol=1e-6):
    """Find line of intersection between two planes"""
    d_vec = np.cross(n1, n2)
    d_norm = norm(d_vec)
    if d_norm < tol:
        return None, None
    d_vec = d_vec / d_norm
    
    # Create coefficient matrix
    A = np.vstack((n1, n2))
    b = np.array([d1, d2])
    
    # Find a point by setting z=0
    if abs(A[0, 2]) > tol or abs(A[1, 2]) > tol:
        A_xy = A[:, :2]
        if np.linalg.det(A_xy) > tol:
            x0, y0 = np.linalg.solve(A_xy, b)
            return np.array([x0, y0, 0]), d_vec
    
    # Set y=0
    if abs(A[0, 1]) > tol or abs(A[1, 1]) > tol:
        A_xz = A[:, [0, 2]]
        if np.linalg.det(A_xz) > tol:
            x0, z0 = np.linalg.solve(A_xz, b)
            return np.array([x0, 0, z0]), d_vec
    
    # Set x=0
    A_yz = A[:, 1:]
    if np.linalg.det(A_yz) > tol:
        y0, z0 = np.linalg.solve(A_yz, b)
        return np.array([0, y0, z0]), d_vec
    
    return None, None

def plot_plane(ax, point, normal, color, size=1.0, alpha=0.2):
    """Plot plane given point and normal vector"""
    normal = normal / norm(normal)
    
    # Find two vectors perpendicular to normal
    if abs(normal[0]) > 1e-6 or abs(normal[1]) > 1e-6:
        perp1 = np.array([-normal[1], normal[0], 0])
    else:
        perp1 = np.array([1, 0, 0])
    perp1 = perp1 / norm(perp1)
    perp2 = np.cross(normal, perp1)
    perp2 = perp2 / norm(perp2)
    
    # Generate grid
    u, v = np.meshgrid(np.linspace(-size, size, 10), 
                       np.linspace(-size, size, 10))
    x = point[0] + perp1[0]*u + perp2[0]*v
    y = point[1] + perp1[1]*u + perp2[1]*v
    z = point[2] + perp1[2]*u + perp2[2]*v
    
    ax.plot_surface(x, y, z, color=color, alpha=alpha)





















def compute_distance_metrics(points1, points2, s_values1, s_values2):
    """
    Compute distance metrics between corresponding intersection points
    
    Returns:
    d_plus_sq: Squared distance between first solutions
    d_minus_sq: Squared distance between second solutions
    asymmetry: Asymmetry ratio |d_plus_sq - d_minus_sq| / mean
    """
    # Check if we have two solutions for each ellipse
    if len(points1) < 2 or len(points2) < 2:
        return None, None, None
    
    # Compute squared distances
    d_plus_sq = norm(points1[0] - points2[0])**2
    d_minus_sq = norm(points1[1] - points2[1])**2
    
    # Compute asymmetry ratio
    mean_sq = 0.5 * (d_plus_sq + d_minus_sq)
    asymmetry = abs(d_plus_sq - d_minus_sq) / mean_sq if mean_sq > 1e-6 else 0
    
    return d_plus_sq, d_minus_sq, asymmetry

if __name__ == "__main__":
    # Use your explicit parameters
    Z1, t1, w1, beta_mu1 = 1.2, 0.5, 0.3, 0.9
    Z2, t2, w2, beta_mu2 = 1.2, 0.4, 0.4, 0.85
    
    # Rotation matrices
    theta1 = np.pi/5
    R_T1 = np.array([
        [1, 0, 0],
        [0, np.cos(theta1), -np.sin(theta1)],
        [0, np.sin(theta1), np.cos(theta1)]
    ])
    
    theta2 = np.pi/4
    R_T2 = np.array([
        [np.cos(theta2), 0, -np.sin(theta2)],
        [0, 1, 0],
        [np.sin(theta2), 0, np.cos(theta2)]
    ])
    
    # Create ellipses
    ellipse1 = KinematicEllipse(Z1, t1, w1, beta_mu1, R_T1)
    ellipse2 = KinematicEllipse(Z2, t2, w2, beta_mu2, R_T2)
    
    # Get plane equations
    n1, d1 = ellipse1.plane_equation()
    n2, d2 = ellipse2.plane_equation()
    
    # Find intersection line
    r0, d_vec = find_intersection_line(n1, d1, n2, d2)
    
    if r0 is None:
        print("Planes are parallel, no intersection line")
    else:
        # Find line-ellipse intersections
        phi1_vals, points1, s_values1 = ellipse1.find_line_intersections(r0, d_vec)
        phi2_vals, points2, s_values2 = ellipse2.find_line_intersections(r0, d_vec)
        
        # Generate points on the line
        s_vals = np.linspace(-2, 2, 100)
        line_pts = np.array([r0 + s*d_vec for s in s_vals])
        
        # Generate ellipse points
        ellipse1_pts = ellipse1.generate_ellipse()
        ellipse2_pts = ellipse2.generate_ellipse()
        
        # Compute distance metrics
        d_plus_sq, d_minus_sq, asymmetry = compute_distance_metrics(points1, points2, s_values1, s_values2)
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot ellipses
        ax.plot(ellipse1_pts[:, 0], ellipse1_pts[:, 1], ellipse1_pts[:, 2], 
                'b-', linewidth=2, label=f'Ellipse 1 (Z={Z1}, t={t1})')
        ax.plot(ellipse2_pts[:, 0], ellipse2_pts[:, 1], ellipse2_pts[:, 2], 
                'r-', linewidth=2, label=f'Ellipse 2 (Z={Z2}, t={t2})')
        
        # Plot centers
        ax.scatter(ellipse1.center[0], ellipse1.center[1], ellipse1.center[2], 
                   s=80, c='blue', marker='o', label='Center 1')
        ax.scatter(ellipse2.center[0], ellipse2.center[1], ellipse2.center[2], 
                   s=80, c='red', marker='o', label='Center 2')
        
        # Plot planes
        plot_plane(ax, ellipse1.center, n1, 'blue', size=1.0)
        plot_plane(ax, ellipse2.center, n2, 'red', size=1.0)
        
        # Plot normals
        ax.quiver(ellipse1.center[0], ellipse1.center[1], ellipse1.center[2],
                  n1[0], n1[1], n1[2], length=0.3, color='darkblue', 
                  linewidth=1.5, label='Normal 1')
        ax.quiver(ellipse2.center[0], ellipse2.center[1], ellipse2.center[2],
                  n2[0], n2[1], n2[2], length=0.3, color='darkred', 
                  linewidth=1.5, label='Normal 2')
        
        # Plot intersection line
        ax.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2], 
                'g-', linewidth=2, label='Intersection Line')
        
        # Plot line-ellipse intersection points
        colors = ['cyan', 'magenta']
        markers = ['o', 's']
        
        # For ellipse 1
        for i, point in enumerate(points1):
            # Draw vector from center to point
            ax.quiver(ellipse1.center[0], ellipse1.center[1], ellipse1.center[2],
                      point[0]-ellipse1.center[0], 
                      point[1]-ellipse1.center[1], 
                      point[2]-ellipse1.center[2],
                      color=colors[i], linewidth=1.5, arrow_length_ratio=0.1)
            
            # Mark intersection point
            ax.scatter(point[0], point[1], point[2], s=100, c=colors[i], marker=markers[i])
            
            # Label phi value
            ax.text(point[0], point[1], point[2], 
                    f'φ1_{i+1}={phi1_vals[i]:.2f}\ns={s_values1[i]:.2f}', 
                    color='black', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.7))
        
        # For ellipse 2
        for i, point in enumerate(points2):
            # Draw vector from center to point
            ax.quiver(ellipse2.center[0], ellipse2.center[1], ellipse2.center[2],
                      point[0]-ellipse2.center[0], 
                      point[1]-ellipse2.center[1], 
                      point[2]-ellipse2.center[2],
                      color=colors[i], linewidth=1.5, arrow_length_ratio=0.1)
            
            # Mark intersection point
            ax.scatter(point[0], point[1], point[2], s=100, c=colors[i], marker=markers[i])
            
            # Label phi value
            ax.text(point[0], point[1], point[2], 
                    f'φ2_{i+1}={phi2_vals[i]:.2f}\ns={s_values2[i]:.2f}', 
                    color='black', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.7))
        
        # Plot distance lines if we have two points per ellipse
        if len(points1) == 2 and len(points2) == 2:
            # Distance between first solutions (d_plus)
            ax.plot([points1[0][0], points2[0][0]], 
                    [points1[0][1], points2[0][1]], 
                    [points1[0][2], points2[0][2]], 
                    'm--', linewidth=2, label=f'd²⁺ = {d_plus_sq:.4f}')
            
            # Distance between second solutions (d_minus)
            ax.plot([points1[1][0], points2[1][0]], 
                    [points1[1][1], points2[1][1]], 
                    [points1[1][2], points2[1][2]], 
                    'c--', linewidth=2, label=f'd²⁻ = {d_minus_sq:.4f}')
        
        # Configure plot
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.set_title('3D Ellipse Intersection Analysis with Distance Metrics', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True)
        
        # Set equal aspect ratio
        all_pts = np.vstack((ellipse1_pts, ellipse2_pts, line_pts))
        if len(points1) > 0:
            all_pts = np.vstack((all_pts, points1))
        if len(points2) > 0:
            all_pts = np.vstack((all_pts, points2))
        
        max_val = np.max(np.abs(all_pts)) * 1.2
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_zlim(-max_val, max_val)
        
        plt.tight_layout()
        plt.show()
        

def compute_line_intersection(N1, C1, N2, C2, tol=1e-8):
    d = np.cross(N1, N2)
    d_norm = np.linalg.norm(d)

    N1_dot_N1 = np.dot(N1, N1)
    N2_dot_N2 = np.dot(N2, N2)
    N1_dot_N2 = np.dot(N1, N2)
    a1 = np.dot(N1, C1)
    a2 = np.dot(N2, C2)

    denominator = N1_dot_N1 * N2_dot_N2 - N1_dot_N2**2
    if abs(denominator) == 0: return None, None, "Planes are parallel"
    r0  = ((a1 * N2_dot_N2 - a2 * N1_dot_N2) * N1 + (a2 * N1_dot_N1 - a1 * N1_dot_N2) * N2)
    r0 /= denominator

    return r0, d, ""

class Ellipse3D:
    def __init__(self, Z, w, beta_mu, Omega, R_T=np.eye(3), name=""):
        self.Z = Z
        self.w = w
        self.beta_mu = beta_mu
        self.Omega = Omega
        self.R_T = R_T
        self.name = name
        self.update_parameters()

    def update_parameters(self):
        # Basis vectors
        self.A_vec = (self.Z/self.Omega) * self.R_T @ np.array([1, self.w, 0])
        self.B_vec = self.Z * self.R_T @ np.array([0, 0, 1])

        # Center components
        denom = np.sqrt(1 + self.w**2)
        self.C1 = (-self.Z*self.beta_mu/(self.Omega*denom)) * self.R_T @ np.array([1, self.w, 0])
        self.C2 = (self.Z/denom) * self.R_T @ np.array([-self.w, 1, 0])

        # Normal vector to the plane
        self.normal = np.cross(self.A_vec, self.B_vec)

        # Precompute magnitudes
        self.a = np.linalg.norm(self.A_vec)**2
        self.b = np.linalg.norm(self.B_vec)**2

    def center(self, t):
        return self.C1 * np.cosh(t) + self.C2 * np.sinh(t)

    def points(self, t, n_points=100):
        C = self.center(t)
        return get_ellipse_points(self.A_vec, self.B_vec, C, n_points)

    def get_point_at_phi(self, phi, t):
        C = self.center(t)
        return C + self.A_vec * np.cos(phi) + self.B_vec * np.sin(phi)

    def solve_intersection(self, r0, d, t):
        if r0 is None or d is None: return [], []

        C = self.center(t)

        # Projection coefficients
        beta  = np.dot(d, self.A_vec) / self.a
        delta = np.dot(d, self.B_vec) / self.b
        alpha = np.dot(r0 - C, self.A_vec) / self.a
        gamma = np.dot(r0 - C, self.B_vec) / self.b

        # Quadratic equation coefficients
        A_coeff = beta**2 + delta**2
        B_coeff = 2 * (alpha*beta + gamma*delta)
        C_coeff = alpha**2 + gamma**2 - 1

        # Solve quadratic
        discriminant = B_coeff**2 - 4*A_coeff*C_coeff
        if discriminant < 0: return [], []
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

        return phi_vals, points

# =============================================
# Intersection Optimization
# =============================================
def find_intersection_parameters(initial_params, fixed_params, max_iter=50):
    """
    Find parameters (t1, t2, Z1, Z2) that minimize distance between ellipses
    while ensuring they intersect
    """
    # Unpack fixed parameters
    w1, beta_mu1, Omega1, w2, beta_mu2, Omega2, R_T1, R_T2 = fixed_params

    # Create rotation matrices if not provided
    if R_T1 is None: R_T1 = rotation_matrix([1, 0, 0], np.pi/6)
    if R_T2 is None: R_T2 = rotation_matrix([0, 1, 0], np.pi/4)

    ellipse1 = Ellipse3D(Z=initial_params[2], w=w1, beta_mu=beta_mu1, Omega=Omega1, R_T=R_T1, name="Ellipse 1")
    ellipse2 = Ellipse3D(Z=initial_params[3], w=w2, beta_mu=beta_mu2, Omega=Omega2, R_T=R_T2, name="Ellipse 2")
    history = []

    # Define the objective function (distance to minimize)
    def objective_function(params):
        t1, t2, Z1, Z2 = params

        # Update ellipse parameters
        ellipse1.Z = Z1
        ellipse2.Z = Z2
        ellipse1.update_parameters()
        ellipse2.update_parameters()

        # Compute centers
        C1 = ellipse1.center(t1)
        C2 = ellipse2.center(t2)

        # Compute intersection line
        r0, d, status = compute_line_intersection(ellipse1.normal, C1, ellipse2.normal, C2)

        # Get intersection points
        _, points1 = ellipse1.solve_intersection(r0, d, t1)
        _, points2 = ellipse2.solve_intersection(r0, d, t2)

        # Calculate minimal distance between ellipses
        min_distance = 1000  # Large default value
        if points1 and points2:
            # Compute minimal distance between any pair of points
            min_distance = min(
                np.linalg.norm(p1 - p2)
                for p1 in points1
                for p2 in points2
            )

        # Store iteration data
        history.append({
            'params': params.copy(),
            'min_distance': min_distance,
            'points1': points1,
            'points2': points2
        })

        return min_distance

    # Run optimization
    res = minimize(objective_function, initial_params, method='L-BFGS-B', options={'maxiter': max_iter})
    t1_opt, t2_opt, Z1_opt, Z2_opt = res.x

    # Final configuration with optimal parameters
    ellipse1.Z = Z1_opt
    ellipse2.Z = Z2_opt
    ellipse1.update_parameters()
    ellipse2.update_parameters()

    # Compute final centers and intersection
    C1_opt = ellipse1.center(t1_opt)
    C2_opt = ellipse2.center(t2_opt)
    r0_opt, d_opt, status = compute_line_intersection(ellipse1.normal, C1_opt, ellipse2.normal, C2_opt)
    phi1_opt, points1_opt = ellipse1.solve_intersection(r0_opt, d_opt, t1_opt)
    phi2_opt, points2_opt = ellipse2.solve_intersection(r0_opt, d_opt, t2_opt)

    print(d_opt, r0_opt)
    return {
        'ellipse1': ellipse1,
        'ellipse2': ellipse2,
        't1': t1_opt,
        't2': t2_opt,
        'Z1': Z1_opt,
        'Z2': Z2_opt,
        'min_distance': res.fun,
        'points1': points1_opt,
        'points2': points2_opt,
        'r0': r0_opt,
        'd': d_opt,
        'history': history
    }

# =============================================
# Visualization and Analysis
# =============================================
def visualize_optimal_solution(result):
    """Create 3D visualization of optimal configuration"""
    ellipse1 = result['ellipse1']
    ellipse2 = result['ellipse2']
    t1 = result['t1']
    t2 = result['t2']

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot ellipses
    ellipse1_pts = ellipse1.points(t1)
    ellipse2_pts = ellipse2.points(t2)
    ax.plot(ellipse1_pts[:,0], ellipse1_pts[:,1], ellipse1_pts[:,2],
            'b-', label=ellipse1.name, linewidth=2, alpha=0.8)
    ax.plot(ellipse2_pts[:,0], ellipse2_pts[:,1], ellipse2_pts[:,2],
            'r-', label=ellipse2.name, linewidth=2, alpha=0.8)

    # Plot line of intersection
    if result['r0'] is not None and result['d'] is not None:
        s_vals = np.linspace(-2, 2, 100)
        line_points = np.array([result['r0'] + s*result['d'] for s in s_vals])
        ax.plot(line_points[:,0], line_points[:,1], line_points[:,2],
                'g-', label='Intersection Line', linewidth=3, alpha=0.7)

    # Plot centers
    C1 = ellipse1.center(t1)
    C2 = ellipse2.center(t2)
    ax.scatter(C1[0], C1[1], C1[2], s=120, c='blue', marker='*',
              edgecolor='k', label=f'{ellipse1.name} Center')
    ax.scatter(C2[0], C2[1], C2[2], s=120, c='red', marker='*',
              edgecolor='k', label=f'{ellipse2.name} Center')

    # Plot intersection points
    if result['points1']:
        pts1 = np.array(result['points1'])
        ax.scatter(pts1[:,0], pts1[:,1], pts1[:,2], s=120, c='cyan',
                   marker='o', edgecolor='k', label=f'{ellipse1.name} Intersections')
    if result['points2']:
        pts2 = np.array(result['points2'])
        ax.scatter(pts2[:,0], pts2[:,1], pts2[:,2], s=120, c='magenta',
                   marker='s', edgecolor='k', label=f'{ellipse2.name} Intersections')

    # Highlight closest points
    if result['points1'] and result['points2']:
        min_distance = result['min_distance']
        min_pair = None

        # Find closest pair of points
        for i, p1 in enumerate(result['points1']):
            for j, p2 in enumerate(result['points2']):
                dist = np.linalg.norm(p1 - p2)
                if abs(dist - min_distance) < 1e-5:
                    min_pair = (p1, p2)
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                            'm--', linewidth=3, alpha=0.8, label=f'Min Distance: {min_distance:.4f}')

        # If we found a pair, add label at midpoint
        if min_pair:
            mid_point = (min_pair[0] + min_pair[1]) / 2
            ax.text(mid_point[0], mid_point[1], mid_point[2],
                    f'd = {min_distance:.4f}', fontsize=12,
                    bbox=dict(facecolor='yellow', alpha=0.8))

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'Optimal Ellipse Intersection: d_min = {result["min_distance"]:.6f}', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_optimization_history(history):
    """Plot optimization progress"""
    iterations = range(len(history))
    distances = [h['min_distance'] for h in history]

    plt.figure(figsize=(12, 6))
    plt.plot(iterations, distances, 'bo-')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Minimal Distance', fontsize=12)
    plt.title('Optimization Progress: Minimal Distance vs Iteration', fontsize=14)
    plt.grid(True)
    plt.show()

# =============================================
# Main Execution
# =============================================
if __name__ == "__main__":
    # Fixed parameters for both ellipses (w, beta_mu, Omega, rotation matrix)
    w1, beta_mu1, Omega1 = 0.2, 0.5, 1.0
    w2, beta_mu2, Omega2 = 0.3, 0.6, 1.1

    # Create rotation matrices
    R_T1 = rotation_matrix([1, 0, 0], np.pi/6)
    R_T2 = rotation_matrix([0, 1, 0], np.pi/4)

    # Initial parameters [t1, t2, Z1, Z2]
    initial_params = [0.5, 0.3, 1.0, 1.2]

    # Find optimal parameters for intersection
    result = find_intersection_parameters(
        initial_params=initial_params,
        fixed_params=[w1, beta_mu1, Omega1, w2, beta_mu2, Omega2, R_T1, R_T2],
        max_iter=50
    )

    # Print results
    print("\n" + "="*70)
    print("Optimal Parameters for Ellipse Intersection")
    print("="*70)
    print(f"Minimal distance achieved: {result['min_distance']:.6f}")
    print(f"Hyperbolic parameters: t1 = {result['t1']:.6f}, t2 = {result['t2']:.6f}")
    print(f"Reconstructed masses: Z1 = {result['Z1']:.6f}, Z2 = {result['Z2']:.6f}")
    print("="*70)

    # Visualization
    visualize_optimal_solution(result)
    plot_optimization_history(result['history'])

    # Print details about intersection points
    if result['points1'] and result['points2']:
        print("\nIntersection Points:")
        print(f"Ellipse 1: {len(result['points1'])} points")
        for i, p in enumerate(result['points1']):
            print(f"  Point {i+1}: {p}")

        print(f"Ellipse 2: {len(result['points2'])} points")
        for i, p in enumerate(result['points2']):
            print(f"  Point {i+1}: {p}")

        # Find and print closest points
        min_distance = result['min_distance']
        for i, p1 in enumerate(result['points1']):
            for j, p2 in enumerate(result['points2']):
                dist = np.linalg.norm(p1 - p2)
                if abs(dist - min_distance) < 1e-5:
                    print(f"\nClosest points (distance = {min_distance:.6f}):")
                    print(f"  From Ellipse 1: Point {i+1} at {p1}")
                    print(f"  From Ellipse 2: Point {j+1} at {p2}")
    else:
        print("\nNo intersection points found - optimization may not have succeeded")





def plot_covariance_ellipsoid(ax, center, cov, scale=1, n_points=50, alpha=0.2, color='red'):
    """Plot a covariance ellipsoid in 3D."""
    # Eigen decomposition for principal axes
    eigvals, eigvecs = np.linalg.eigh(cov)
    radii = scale * np.sqrt(eigvals)
    
    # Generate points on a sphere
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    # Transform to ellipsoid
    for i in range(len(x)):
        for j in range(len(x)):
            x[i,j], y[i,j], z[i,j] = center + eigvecs @ (radii * np.array([x[i,j], y[i,j], z[i,j]]))
    
    ax.plot_surface(x, y, z, alpha=alpha, color=color)

def main(base_path):
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # Read included systems
    included = []
    try:
        with open(f"{base_path}_included.txt", 'r') as f:
            included = [int(line.strip()) for line in f]
    except FileNotFoundError:
        print("Included systems file not found. Plotting all systems.")
        # Try to detect number of systems
        i = 0
        while os.path.exists(f"{base_path}_ellipse_{i}.csv"):
            included.append(i)
            i += 1
    
    # Read and plot common intersection point
    pstar = read_csv(f"{base_path}_pstar.csv")
    ax.scatter(pstar[0], pstar[1], pstar[2], s=200, c='gold', marker='*', 
               edgecolors='black', depthshade=False, label='Intersection Point')
    
    # Read and plot missing energy vector
    p_miss = read_csv(f"{base_path}_missing_energy.csv")
    ax.scatter(p_miss[0], p_miss[1], p_miss[2], s=150, c='purple', marker='d', 
               label='Missing Energy')
    
    # Read covariance matrix and plot uncertainty ellipsoid
    try:
        cov_matrix = read_matrix(f"{base_path}_covariance.csv")
        plot_covariance_ellipsoid(ax, p_miss, cov_matrix, scale=1.0, color='purple')
    except FileNotFoundError:
        print("Covariance matrix file not found. Skipping ellipsoid.")
    
    # Read total neutrino momentum
    total_momentum = read_csv(f"{base_path}_total_momentum.csv")
    
    distances = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(included)))
    
    for idx, i in enumerate(included):
        try:
            # Read ellipse data
            ellipse_points = read_csv(f"{base_path}_ellipse_{i}.csv")
            center = read_csv(f"{base_path}_center_{i}.csv")
            normal = read_csv(f"{base_path}_normal_{i}.csv")
            ellipse_point = read_csv(f"{base_path}_point_{i}.csv")
            distance = float(np.loadtxt(f"{base_path}_distance_{i}.txt"))
            distances.append(distance)
            
            # Plot ellipse
            ax.plot(ellipse_points[:,0], ellipse_points[:,1], ellipse_points[:,2], 
                    color=colors[idx], alpha=0.8, linewidth=2, label=f'Ellipse {i}')
            
            # Plot center (neutrino momentum approximation)
            ax.scatter(center[0], center[1], center[2], s=80, 
                       color=colors[idx], marker='o', alpha=0.8)
            
            # Plot normal vector as arrow
            ellipse_size = np.max(ellipse_points, axis=0) - np.min(ellipse_points, axis=0)
            scale = np.linalg.norm(ellipse_size) * 0.2
            normal_end = center + normal * scale
            
            ax.quiver(center[0], center[1], center[2],
                      normal_end[0]-center[0], normal_end[1]-center[1], normal_end[2]-center[2],
                      color=colors[idx], arrow_length_ratio=0.3, linewidth=2, 
                      length=scale, alpha=0.9)
            
            # Plot closest point
            ax.scatter(ellipse_point[0], ellipse_point[1], ellipse_point[2], 
                       s=100, color=colors[idx], marker='X', alpha=0.9)
            
            # Plot distance line
            ax.plot([pstar[0], ellipse_point[0]], 
                    [pstar[1], ellipse_point[1]], 
                    [pstar[2], ellipse_point[2]], 
                    'k--', alpha=0.7, linewidth=1.5)
                    
        except FileNotFoundError as e:
            print(f"Warning: Missing data for system {i}: {str(e)}")
    
    # Plot total neutrino momentum vector
    ax.quiver(0, 0, 0, 
              total_momentum[0], total_momentum[1], total_momentum[2],
              color='black', linewidth=3, arrow_length_ratio=0.1, 
              label='Total Neutrino Momentum')
    
    # Plot missing energy connection
    ax.plot([p_miss[0], total_momentum[0]], 
            [p_miss[1], total_momentum[1]], 
            [p_miss[2], total_momentum[2]], 
            'm--', linewidth=2, alpha=0.7)
    
    # Add labels and legend
    ax.set_xlabel('X Axis (GeV)', fontsize=12)
    ax.set_ylabel('Y Axis (GeV)', fontsize=12)
    ax.set_zlabel('Z Axis (GeV)', fontsize=12)
    
    title = f'Multi-Neutrino System Solution\n'
    title += f'Distances: {", ".join([f"{d:.2f} GeV" for d in distances])}\n'
    title += f'Total Momentum: ({total_momentum[0]:.1f}, {total_momentum[1]:.1f}, {total_momentum[2]:.1f}) GeV\n'
    title += f'Missing Energy: ({p_miss[0]:.1f}, {p_miss[1]:.1f}, {p_miss[2]:.1f}) GeV'
    
    ax.set_title(title, fontsize=14)
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
               markersize=15, label='Intersection Point'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor='purple', 
               markersize=10, label='Missing Energy'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='Neutrino Momentum'),
        Line2D([0], [0], color='black', linewidth=3, 
               label='Total Neutrino Momentum'),
        Line2D([0], [0], color='m', linestyle='--', linewidth=2, 
               label='Energy Constraint')
    ]
    
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    # Add grid and set view
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Equal aspect ratio
    max_range = np.ptp(np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])).max() / 2.0
    mid_x = (ax.get_xlim()[0] + ax.get_xlim()[1]) * 0.5
    mid_y = (ax.get_ylim()[0] + ax.get_ylim()[1]) * 0.5
    mid_z = (ax.get_zlim()[0] + ax.get_zlim()[1]) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save and show
    plt.tight_layout()
    plt.savefig(f"{base_path}_plot.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_path}_plot.pdf", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plot_ellipses.py <base_path>")
        sys.exit(1)
    base_path = sys.argv[1]
    main(base_path)


import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_random_surface(dim=3):
    """Generate random hyperbolic surface parameters (a_i, b_i)"""
    return (np.random.randn(dim),  # a_i (3x1 vector)
            np.random.randn(dim))  # b_i (3x1 vector)

def surface_point(a, b, t):
    """Compute point on hyperbolic surface H_i(t) = a*sinh(t) + b*cosh(t)"""
    return a * np.sinh(t) + b * np.cosh(t)

def case1_test(i, surfaces, tol=1e-6):
    """Case 1: Check if sum_{j≠i} H_j can be zero"""
    n = len(surfaces)
    other_indices = [j for j in range(n) if j != i]
    
    def objective(t_vec):
        """Objective: ||Σ_{j≠i} (a_j sinh(t_j) + b_j cosh(t_j))||^2"""
        S_i = np.zeros(3)
        for idx, j in enumerate(other_indices):
            a_j, b_j = surfaces[j]
            t_j = t_vec[idx]
            S_i += surface_point(a_j, b_j, t_j)
        return np.linalg.norm(S_i)**2

    # Optimize over t_j for j≠i
    initial_guess = np.zeros(len(other_indices))
    res = minimize(objective, initial_guess, method='L-BFGS-B')
    
    return res.fun < tol, res

def case2_test(i, surfaces, tol=1e-6):
    """Case 2: Full optimization for s, t_i, and {t_j}"""
    n = len(surfaces)
    other_indices = [j for j in range(n) if j != i]
    a_i, b_i = surfaces[i]
    
    def objective(x):
        """Objective: ||u(s,t_i) - Σ_{j≠i} H_j(t_j)||^2"""
        s = x[0]
        t_i = x[1]
        t_other = x[2:]
        
        # Compute left side u(s, t_i)
        u = (a_i * (np.sinh(s) - np.sinh(t_i)) + 
             b_i * (np.cosh(s) - np.cosh(t_i)))
        
        # Compute right side S_i = Σ_{j≠i} H_j(t_j)
        S_i = np.zeros(3)
        for idx, j in enumerate(other_indices):
            a_j, b_j = surfaces[j]
            S_i += surface_point(a_j, b_j, t_other[idx])
        print(S_i - u)
        return np.linalg.norm(u - S_i)**2

    # Optimize over s, t_i, and t_j for j≠i
    initial_guess = np.zeros(1 + 1 + len(other_indices))
    res = minimize(objective, initial_guess, method='L-BFGS-B')
    
    return res.fun < tol, res

def visualize_surfaces(surfaces, i, res_case1, res_case2):
    """Visualize surfaces and intersection test results"""
    fig = plt.figure(figsize=(15, 10))
    
    # Plot all surfaces
    ax1 = fig.add_subplot(121, projection='3d')
    t_vals = np.linspace(-2, 2, 10)
    
    for idx, (a, b) in enumerate(surfaces):
        points = np.array([surface_point(a, b, t) for t in t_vals])
        ax1.plot(points[:,0], points[:,1], points[:,2], 
                label=f'Surface {idx}', alpha=0.7 if idx != i else 1.0,
                linewidth=3 if idx == i else 1)
    
    ax1.set_title(f'All {len(surfaces)} Hyperbolic Surfaces')
    ax1.legend()
    
    # Plot intersection test results
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Intersection Analysis')
    
    # Plot i-th surface (thicker line)
    points_i = np.array([surface_point(*surfaces[i], t) for t in t_vals])
    ax2.plot(points_i[:,0], points_i[:,1], points_i[:,2], 'r-', linewidth=3, label=f'Target Surface {i}')
    total_points = []
    for t_combo in np.array(np.meshgrid(*[t_vals]*(len(surfaces)))).T.reshape(-1, len(surfaces)):
        point_sum = sum(surface_point(a, b, t) for (a, b), t in zip(surfaces, t_combo))
        total_points.append(point_sum)
    total_points = np.array(total_points)
    
    ax2.scatter(total_points[:,0], total_points[:,1], total_points[:,2], alpha=0.1, s=1, label='Total Surface Points')
    
    # Highlight intersection points if found
    if res_case1[0]:
        ax2.scatter(0, 0, 0, c='g', s=200, marker='*', label='Case 1 Intersection (S_i=0)')
    
    if res_case2[0]:
        # Extract solution point
        s_opt = res_case2[1].x[0]
        point_intersect = surface_point(*surfaces[i], s_opt)
        ax2.scatter(*point_intersect, c='b', s=200, marker='X', label='Case 2 Intersection')
    
    ax2.legend()
    plt.tight_layout()
    plt.show()

# Main script
np.random.seed(42)

# Generate n random hyperbolic surfaces
n = 3  # Number of surfaces
surfaces = [generate_random_surface() for _ in range(n)]
i = 0  # Test intersection for i-th surface

# Run intersection tests
case1_result, case1_res = case1_test(i, surfaces)
case2_result, case2_res = case2_test(i, surfaces)

print(f"Case 1 (S_i = 0 possible): {case1_result} | Residual: {case1_res.fun:.6f}")
print(f"Case 2 (Full solution found): {case2_result} | Residual: {case2_res.fun:.6f}")

# Visualization
visualize_surfaces(surfaces, i, (case1_result, case1_res), (case2_result, case2_res))

