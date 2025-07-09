import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


# Verification: Apply rotation to initial vector
def rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle/2)
    b, c, d = -axis * np.sin(angle/2)
    return np.array([
        [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]
    ])


# Function to find rotation angles
def find_rotation_angles(u0, target_normal):
    # Normalize the target normal
    n_target = target_normal / np.linalg.norm(target_normal)
    
    # Calculate the rotation axis (perpendicular to both vectors)
    rot_axis = np.cross(u0, n_target)
    rot_axis_norm = np.linalg.norm(rot_axis)
    
    # If vectors are parallel, return zero angles
    if rot_axis_norm < 1e-10: return 0.0, 0.0
    
    # Normalize rotation axis
    rot_axis /= rot_axis_norm
    
    # Calculate rotation angle
    cos_angle = np.dot(u0, n_target)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    # Convert rotation axis and angle to spherical coordinates
    x, y, z = rot_axis
    phi = np.arccos(z)  # Polar angle
    theta = np.arctan2(y, x)  # Azimuthal angle
    
    return theta, phi, angle

def compute_inscribed_circle(ellipse):
    A = ellipse[:, 0]
    B = ellipse[:, 1]
    C = ellipse[:, 2]
    
    M00 = np.dot(A, A)
    M01 = np.dot(A, B)
    M11 = np.dot(B, B)
    
    trace = M00 + M11
    det = M00*M11 - M01**2
    discriminant = np.sqrt(trace**2 - 4*det)
    lambda1 = (trace + discriminant)/2
    lambda2 = (trace - discriminant)/2
    
    radius = np.sqrt(min(lambda1, lambda2))
    
    u = A / np.linalg.norm(A)
    v_temp = B - np.dot(B, u) * u
    v = v_temp / np.linalg.norm(v_temp)
    
    t = np.linspace(0, 2*np.pi, 1000)
    circle = np.outer(np.cos(t), u) + np.outer(np.sin(t), v)
    circle = radius * circle + C
    
    return circle, radius

def ellipse_properties(ellipse):
    """Compute geometric properties of a 3D ellipse"""
    A = ellipse[:, 0]
    B = ellipse[:, 1]
    C = ellipse[:, 2]  # Centroid
    
    # Compute normal vector
    N = np.cross(A, B)
    norm_N = np.linalg.norm(N)
    if norm_N > 1e-10:
        N_normalized = N / norm_N
    else:
        N_normalized = N
    
    # Compute metric tensor
    M = np.array([[np.dot(A, A), np.dot(A, B)], 
                  [np.dot(A, B), np.dot(B, B)]])
    
    # Compute eigenvalues
    trace = M[0,0] + M[1,1]
    det = M[0,0]*M[1,1] - M[0,1]**2
    discriminant = np.sqrt(trace**2 - 4*det)
    lambda1 = (trace + discriminant)/2
    lambda2 = (trace - discriminant)/2
    
    # Semi-axis lengths
    semi_major = np.sqrt(max(lambda1, lambda2))
    semi_minor = np.sqrt(min(lambda1, lambda2))
    area = np.pi * semi_major * semi_minor
    
    return {
        'centroid': C,
        'normal': N_normalized,
        'semi_major': semi_major,
        'semi_minor': semi_minor,
        'area': area
    }

def plane_equation(params, points):
    a, b, c, d = params
    x, y, z = points.T
    return np.abs(a*x + b*y + c*z + d) / np.sqrt(a**2 + b**2 + c**2)

def fit_plane(points):
    centroid = np.mean(points, axis=0)
    def residuals(params):
        return plane_equation(params, points - centroid)
    
    result = least_squares(residuals, [1, 1, 1, 0])
    a, b, c, d = result.x
    normal = np.array([a, b, c])
    norm = np.linalg.norm(normal)
    if norm > 1e-10:
        normal /= norm
        d = d / norm + np.dot(normal, centroid)
    return normal, d

def UnitCircle(): return np.diag([1, 1, -1])

def find_intersection_points(ellipse1, ellipse2, num_points=1000, tol=1e-3):
    t = np.linspace(0, 2*np.pi, num_points)
    
    # Generate points for both ellipses
    pts1 = np.array([
        ellipse1[0,0]*np.cos(t) + ellipse1[0,1]*np.sin(t) + ellipse1[0,2],
        ellipse1[1,0]*np.cos(t) + ellipse1[1,1]*np.sin(t) + ellipse1[1,2],
        ellipse1[2,0]*np.cos(t) + ellipse1[2,1]*np.sin(t) + ellipse1[2,2]
    ]).T
    
    pts2 = np.array([
        ellipse2[0,0]*np.cos(t) + ellipse2[0,1]*np.sin(t) + ellipse2[0,2],
        ellipse2[1,0]*np.cos(t) + ellipse2[1,1]*np.sin(t) + ellipse2[1,2],
        ellipse2[2,0]*np.cos(t) + ellipse2[2,1]*np.sin(t) + ellipse2[2,2]
    ]).T
    
    # Find closest points between the two ellipses
    dist_matrix = cdist(pts1, pts2)
    
    # Find pairs of points closer than tolerance
    intersection_points = []
    for i in range(num_points):
        min_dist = np.min(dist_matrix[i])
        if min_dist < tol:
            j = np.argmin(dist_matrix[i])
            # Take midpoint as approximate intersection
            intersection_points.append((pts1[i] + pts2[j])/2)
    
    return np.array(intersection_points)

# Define the ellipse matrices
ellipse1 = np.array([
    [-90932.3243752, -132823.40737144, -81627.03290768],
    [-246713.23558281, 39185.25049644, -224263.71794771],
    [159277.4178682, -15133.47754428, 135735.5355611]
])

ellipse2 = np.array([
    [42648.05657968, -30585.43118447, 75941.69144114],
    [-125535.91423418, -27323.23855879, -154229.10126757],
    [-81801.14518129, 25985.43734243, -72784.19539811]
])

# Define the circle matrix
#circle_matrix = np.array([
#[-1.00000000e+00,  0.00000000e+00,  5.32179206e+04],
#[ 0.00000000e+00, -1.00000000e+00, -7.06466657e+04],
#[ 5.32179206e+04, -7.06466657e+04,  1.00000000e+00],
#])

v0 = np.array([106435.84125419, -141293.33148039, 0.0])
_alpha = np.radians(-143.009395*0 + 0*63.587642 + 1*106.684975) 
_gamma = np.radians(-143.009395*0 + 1*63.587642 + 0*106.684975) #63.587642)
_beta  = np.radians(-143.009395*0 + 0*63.587642 + 0*106.684975)
ca, sa = np.cos(_alpha), np.sin(_alpha)
cg, sg = np.cos(_gamma), np.sin(_gamma)
cb, sb = np.cos(_beta) , np.sin(_beta)

R = np.array([
    [cb*cg, sa*sb*cg - ca*sg, ca*sb*cg + sa*sg], 
    [cb*sg, sa*sb*sg + ca*cg, ca*sb*sg - sa*cg], 
    [-sb  , sa*cb           , ca*cb           ]
])
T = R #np.array([[R[0][0], R[0][1], R[0][2]], [, R[1][1], 0], [0, 0, 1]])

S = np.outer(v0, [0, 0, 1])#- UnitCircle()
#S = (S + S.T)/2
S = T.T.dot(S).dot(T)
circle_matrix = S
v0 = np.array([S[0][2], S[1][2], S[2][2]*0])
print(S)
print(v0)

# Calculate properties for ellipses
props1 = ellipse_properties(ellipse1)
props2 = ellipse_properties(ellipse2)

# Calculate properties for the circle
circle_props = ellipse_properties(circle_matrix)

# Find intersection points
intersection_pts = find_intersection_points(ellipse1, ellipse2, 2000, tol=50000)
print(f"Found {len(intersection_pts)} approximate intersection points")

# Compute inscribed circles
circle1, radius1 = compute_inscribed_circle(ellipse1)
circle2, radius2 = compute_inscribed_circle(ellipse2)

# Generate points for ellipses
t = np.linspace(0, 2*np.pi, 1000)
x1 = ellipse1[0,0]*np.cos(t) + ellipse1[0,1]*np.sin(t) + ellipse1[0,2]
y1 = ellipse1[1,0]*np.cos(t) + ellipse1[1,1]*np.sin(t) + ellipse1[1,2]
z1 = ellipse1[2,0]*np.cos(t) + ellipse1[2,1]*np.sin(t) + ellipse1[2,2]

x2 = ellipse2[0,0]*np.cos(t) + ellipse2[0,1]*np.sin(t) + ellipse2[0,2]
y2 = ellipse2[1,0]*np.cos(t) + ellipse2[1,1]*np.sin(t) + ellipse2[1,2]
z2 = ellipse2[2,0]*np.cos(t) + ellipse2[2,1]*np.sin(t) + ellipse2[2,2]

# Generate points for the circle
x_circle = circle_matrix[0,0]*np.cos(t) + circle_matrix[0,1]*np.sin(t) + circle_matrix[0,2]
y_circle = circle_matrix[1,0]*np.cos(t) + circle_matrix[1,1]*np.sin(t) + circle_matrix[1,2]
z_circle = circle_matrix[2,0]*np.cos(t) + circle_matrix[2,1]*np.sin(t) + circle_matrix[2,2]

# Fit plane to intersection points
if len(intersection_pts) > 3:
    plane_normal, plane_d = fit_plane(intersection_pts)
    print(f"Plane equation: {plane_normal[0]:.6f}x + {plane_normal[1]:.6f}y + {plane_normal[2]:.6f}z + {plane_d:.6f} = 0")
    
    # Create a grid for the plane visualization
    centroid = np.mean(intersection_pts, axis=0)
    basis1 = np.cross(plane_normal, [1, 0, 0])
    if np.linalg.norm(basis1) < 1e-5: basis1 = np.cross(plane_normal, [0, 1, 0])
    basis1 /= np.linalg.norm(basis1)
    basis2 = np.cross(plane_normal, basis1)
    basis2 /= np.linalg.norm(basis2)
    
    size = 200000
    x_grid, y_grid = np.meshgrid(np.linspace(-size, size, 10), np.linspace(-size, size, 10))
    plane_points = np.zeros((10, 10, 3))
    for i in range(10):
        for j in range(10):
            plane_points[i, j] = centroid + x_grid[i,j]*basis1 + y_grid[i,j]*basis2
else:
    print("Not enough intersection points to define a plane")

# Given initial vector
u0 = v0 / np.linalg.norm(v0)
target_normal = plane_normal

# Calculate rotation angles
theta, phi, rotation_angle = find_rotation_angles(u0, target_normal)

# Convert to degrees for readability
theta_deg = np.degrees(theta)
phi_deg = np.degrees(phi)
rotation_angle_deg = np.degrees(rotation_angle)

# Apply rotation
rot_axis = np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
R = rotation_matrix(rot_axis, rotation_angle)
v_rotated = R.dot(u0)

# Calculate alignment with target normal
alignment = np.dot(v_rotated, target_normal) / (np.linalg.norm(v_rotated) * np.linalg.norm(target_normal))

# Print results
print("\nRotation Results:")
print(f"Initial vector: {v0}")
print(f"Normalized initial vector: {u0}")
print(f"Target normal (plane normal): {target_normal}")
print(f"\nRotation angles:")
print(f"  Theta (azimuthal): {theta:.6f} rad ({theta_deg:.6f} deg)")
print(f"  Phi (polar): {phi:.6f} rad ({phi_deg:.6f} deg)")
print(f"  Rotation angle: {rotation_angle:.6f} rad ({rotation_angle_deg:.6f} deg)")
print(f"\nVerification:")
print(f"  Rotated vector: {v_rotated}")
print(f"  Alignment with target normal: {alignment:.10f} (should be close to 1.0)")

# Additional check: Angle between rotated vector and target normal
angle_diff = np.arccos(np.clip(alignment, -1.0, 1.0))
print(f"  Angle between vectors: {np.degrees(angle_diff):.6f} deg")


# Print circle properties
print("\nCircle Properties:")
print(f"  Centroid: {circle_props['centroid']}")
print(f"  Normal: {circle_props['normal']}")
print(f"  Radius: {circle_props['semi_major']:.2f}")  # Should be same for both axes in a circle
print(f"  Area: {circle_props['area']:.2f}")

# Create plot
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot ellipses
ax.plot(x1, y1, z1, label='Ellipse 1', color='blue', linewidth=2, alpha=0.9)
ax.plot(x2, y2, z2, label='Ellipse 2', color='red' , linewidth=2, alpha=0.9)

# Plot the circle
ax.plot(x_circle, y_circle, z_circle, label='Circle', color='purple', linewidth=3.0, linestyle='-')

# Plot inscribed circles
ax.plot(circle1[:, 0], circle1[:, 1], circle1[:, 2], label='Inscribed Circle 1', color='lightblue', linewidth=2.5, alpha=0.7)
ax.plot(circle2[:, 0], circle2[:, 1], circle2[:, 2], label='Inscribed Circle 2', color='salmon', linewidth=2.5, alpha=0.7)

# Plot centroids
ax.scatter(*props1['centroid'], s=120, c='cyan', marker='o', edgecolor='black', label='Centroid 1')
ax.scatter(*props2['centroid'], s=120, c='yellow', marker='o', edgecolor='black', label='Centroid 2')
ax.scatter(*circle_props['centroid'], s=120, c='magenta', marker='o', edgecolor='black', label='Circle Centroid')

# Plot normal vectors
scale = 50000
arrow1 = Arrow3D(
    [props1['centroid'][0], props1['centroid'][0] + scale*props1['normal'][0]],
    [props1['centroid'][1], props1['centroid'][1] + scale*props1['normal'][1]],
    [props1['centroid'][2], props1['centroid'][2] + scale*props1['normal'][2]],
    mutation_scale=20, lw=2, arrowstyle="-|>", color="darkblue"
)
arrow2 = Arrow3D(
    [props2['centroid'][0], props2['centroid'][0] + scale*props2['normal'][0]],
    [props2['centroid'][1], props2['centroid'][1] + scale*props2['normal'][1]],
    [props2['centroid'][2], props2['centroid'][2] + scale*props2['normal'][2]],
    mutation_scale=20, lw=2, arrowstyle="-|>", color="darkred"
)
arrow_circle = Arrow3D(
    [circle_props['centroid'][0], circle_props['centroid'][0] + scale*circle_props['normal'][0]],
    [circle_props['centroid'][1], circle_props['centroid'][1] + scale*circle_props['normal'][1]],
    [circle_props['centroid'][2], circle_props['centroid'][2] + scale*circle_props['normal'][2]],
    mutation_scale=20, lw=2, arrowstyle="-|>", color="darkviolet"
)
ax.add_artist(arrow1)
ax.add_artist(arrow2)
ax.add_artist(arrow_circle)

# Plot intersection points
if len(intersection_pts) > 0:
    ax.scatter(intersection_pts[:, 0], intersection_pts[:, 1], intersection_pts[:, 2], s=80, c='green', marker='o', label='Intersection Points')

# Plot plane of intersection
if len(intersection_pts) > 3:
    ax.plot_surface(plane_points[:, :, 0], plane_points[:, :, 1], plane_points[:, :, 2], alpha=0.3, color='green', label='Intersection Plane')

# Configure plot
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)
ax.set_title('3D Ellipses with Circle and Intersection Plane', fontsize=14)
ax.legend(fontsize=10, loc='best')
ax.view_init(elev=25, azim=-40)

# Set equal aspect ratio
all_x = np.concatenate([x1, x2, x_circle])
all_y = np.concatenate([y1, y2, y_circle])
all_z = np.concatenate([z1, z2, z_circle])

max_range = np.array([all_x.max()-all_x.min(), 
                      all_y.max()-all_y.min(), 
                      all_z.max()-all_z.min()]).max() * 0.5

mid_x = (all_x.max()+all_x.min()) * 0.5
mid_y = (all_y.max()+all_y.min()) * 0.5
mid_z = (all_z.max()+all_z.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.show()













