from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from classes import *
from atomics import *
import numpy as np

class PencilLine:
    def __init__(self, p0, direction):
        self.p0 = np.array(p0)  # MET point
        self.d = np.array(direction) / np.linalg.norm(direction)
        self.intersections = []
    
    def point_at(self, lambda_val):
        return self.p0 + lambda_val * self.d
    
    def distance_to_point(self, point):
        point_vec = np.array(point) - self.p0
        projection = np.dot(point_vec, self.d)
        closest_point = self.p0 + projection * self.d
        return np.linalg.norm(point - closest_point)

class NeutrinoEllipse:
    def __init__(self, conuic_engine, label = ""):
        self.engine     = conuic_engine
        self.label      = label
        self.center     = None
        self.normal     = None
        self.major_axis = None
        self.minor_axis = None
        self.points_3d  = None
        self.extract_parameters()
    
    def extract_parameters(self):
        H = self.engine.Hmatrix(self.engine.z, self.engine.tau)
        self.center     = H[:, 2]
        self.major_axis = H[:, 0]  # cos(phi) coefficient
        self.minor_axis = H[:, 1]  # sin(phi) coefficient
        self.normal = np.cross(self.major_axis, self.minor_axis)
        self.normal = self.normal / np.linalg.norm(self.normal)
    
    def point_at_angle(self, phi):
        t_vec = np.array([np.cos(phi), np.sin(phi), 1])
        return self.engine.Hmatrix(self.engine.z, self.engine.tau).dot(t_vec)
    
    def find_phi_for_line_intersection(self, line, lambda_val):
        H = self.engine.Hmatrix(self.engine.z, self.engine.tau)
        A = H[:, :2]  
        b = line.point_at(lambda_val) - H[:, 2] 
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        cos_phi, sin_phi = x
        
        norm = np.sqrt(cos_phi**2 + sin_phi**2)
        if norm <= 0: return np.arctan2(sin_phi, cos_phi), residuals
        cos_phi /= norm; sin_phi /= norm
        return np.arctan2(sin_phi, cos_phi), residuals


class MultiNeutrinoSolver:
    def __init__(self, conuic_engines, met_point):
        self.engines = conuic_engines
        self.met = np.array(met_point)
        self.ellipses = [NeutrinoEllipse(engine) for engine in self.engines]
        self.pencil_line = None
        self.solutions = []
    
    def compute_line_ellipse_intersections(self):
        intersections = []
        for i, ellipse in enumerate(self.ellipses):
            numerator = np.dot(ellipse.center - self.pencil_line.p0, ellipse.normal)
            denominator = np.dot(self.pencil_line.d, ellipse.normal)
            if abs(denominator) <= 1e-12: continue
            lambda_intersect = numerator / denominator
            intersection_point = self.pencil_line.point_at(lambda_intersect)
            
            phi, residual = ellipse.find_phi_for_line_intersection(self.pencil_line, lambda_intersect)
            intersections.append({
                'ellipse_idx': i, 'lambda_val': lambda_intersect,
                'point': intersection_point, 'phi': phi, 'residual': residual
            })
        return intersections
    
    def solve_analytically(self):
        avg_normal = np.mean([ellipse.normal for ellipse in self.ellipses], axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        self.pencil_line = PencilLine(self.met, avg_normal)

        intersections = self.compute_line_ellipse_intersections()
        phi_solutions = [(item['ellipse_idx'], item['phi'], item['residual']) for item in intersections]
        self.solutions = []
        for ellipse_idx, phi, residual in phi_solutions:
            ellipse = self.ellipses[ellipse_idx]
            nu_momentum = ellipse.point_at_angle(phi)
            self.solutions.append(nu_momentum)
        return self.solutions
    
    def refine_with_met_constraint(self):
        if not self.solutions: return None
        current_sum = np.sum(self.solutions, axis=0)
        met_error = current_sum[:2] - self.met[:2]  
        adjustment = met_error / len(self.solutions)
        for i in range(len(self.solutions)): self.solutions[i][:2] -= adjustment
        return self.solutions

    def visualize_system(self, show_intersections=True):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, ellipse in enumerate(self.ellipses):
            phi_vals = np.linspace(0, 2*np.pi, 100)
            points = np.array([ellipse.point_at_angle(phi) for phi in phi_vals])
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color=colors[i % len(colors)], label=f'Ellipse {i}', alpha=0.7)
            ax.scatter(*ellipse.center, color=colors[i % len(colors)], marker='o', s=50)
        
        if self.pencil_line:
            lambda_vals = np.linspace(-100, 100, 100)  
            line_points = np.array([self.pencil_line.point_at(l) for l in lambda_vals])
            ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 'k-', linewidth=2, label='Pencil Line')
        
        ax.scatter(*self.met, color='black', marker='*', s=200, label='MET')
        if show_intersections and self.solutions:
            intersections = self.compute_line_ellipse_intersections()
            for item in intersections: ax.scatter(*item['point'], color='cyan', marker='x', s=100)
            for i, solution in enumerate(self.solutions):
                ax.scatter(*solution, color=colors[i % len(colors)], marker='^', s=100, label=f'Nu {i}')



        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)
        ax.set_xlabel('Px')
        ax.set_ylabel('Py') 
        ax.set_zlabel('Pz')
        ax.legend()
        plt.title('Multi-Neutrino Reconstruction System')
        plt.show()

# Example usage function
def run_multi_neutrino_reconstruction(event):
    cx = Conuic(event.met, event.phi, list(event.DetectorObjects.values()), event)
    met_3d = [cx.px, cx.py, cx.pz]
    solver = MultiNeutrinoSolver(cx.engine, met_3d)
    solutions = solver.solve_analytically()
    solver.refine_with_met_constraint()
    
    print(f"Event {event.idx} Solutions:")
    for i, nu in enumerate(solver.solutions): print(f"  Neutrino {i}: Px={nu[0]:.2f}, Py={nu[1]:.2f}, Pz={nu[2]:.2f}")
    solver.visualize_system()
    return solver.solutions

# Main execution
if __name__ == "__main__":
    data_loader = DataLoader()
    for event in data_loader:
        #if event.idx != 18: continue
        solutions = run_multi_neutrino_reconstruction(event)
  #      exit()
