import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import AnchoredText

class ellipse:
    def __init__(self, col=None, line=None, label=None, H_tilde=None, sols=None, mrk=None):
        self.col = col
        self.line = line
        self.H = H_tilde
        self.label = label
        self.sol_pts = sols
        self.points = None
        self.mrk_pts = mrk
        self.bounds = None  # Will store min/max bounds

    def compile(self, t_values):
        if self.H is None:
            return
        self.points = np.array([self.H.dot(np.array([np.cos(t), np.sin(t), 1])) for t in t_values])
        # Calculate bounds
        if self.points is not None and len(self.points) > 0:
            self.bounds = {
                'x': (np.min(self.points[:, 0]), np.max(self.points[:, 0])),
                'y': (np.min(self.points[:, 1]), np.max(self.points[:, 1])),
                'z': (np.min(self.points[:, 2]), np.max(self.points[:, 2]))
            }

class hyperbolic:
    def __init__(self, col=None, line=None, label=None, fx=None, domain=None, pts=None):
        self.col = col
        self.label = label
        self.line = line
        self.fx = fx
        self.H = None
        self.raw_points = None
        self.points = None  # Will hold clipped points
        self.param_values = np.linspace(domain[0], domain[1], 1000)
        self.sol_pts = pts
        self.mrk_pts = None
        self.domain = domain

    def compile(self, t_values=None):
        if self.fx is None: return
        self.raw_points = np.array([self.fx(t) for t in self.param_values])

    def clip_to_bounds(self, bounds, margin_factor=1.2):
        """Clip hyperbolic curve to reasonable bounds based on ellipse data"""
        if self.raw_points is None or bounds is None:
            self.points = self.raw_points
            return

        # Get bounds with margin
        x_min, x_max = bounds['x']
        y_min, y_max = bounds['y']
        z_min, z_max = bounds['z']
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        x_min_clip = x_min - margin_factor * x_range
        x_max_clip = x_max + margin_factor * x_range
        y_min_clip = y_min - margin_factor * y_range
        y_max_clip = y_max + margin_factor * y_range
        z_min_clip = z_min - margin_factor * z_range
        z_max_clip = z_max + margin_factor * z_range

        # Find points within bounds
        mask = (
            (self.raw_points[:, 0] >= x_min_clip) & (self.raw_points[:, 0] <= x_max_clip) &
            (self.raw_points[:, 1] >= y_min_clip) & (self.raw_points[:, 1] <= y_max_clip) &
            (self.raw_points[:, 2] >= z_min_clip) & (self.raw_points[:, 2] <= z_max_clip)
        )
        
        self.points = self.raw_points[mask]
        if len(self.points) < 10: self.points = self.raw_points

class packet:
    def __init__(self, truth=None, tru_pts=None, pts=1000, inst=None):
        self.truth = ellipse("red", "solid", 'Truth Neutrino', truth, tru_pts, "*")
        self.reco = []
        self.pts = pts
        self.instance = inst
        
        self.cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        self.style = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
        self.pts_ = ["o", "s", "^", "v", "D", "P", "X"]
        
        # Create iterators
        self.cols_iter = None
        self.style_iter = None
        self.pts_iter = None
    
    def _reset_iterators(self):
        self.cols_iter = itertools.cycle(self.cols)
        self.style_iter = itertools.cycle(self.style)
        self.pts_iter = itertools.cycle(self.pts_)
    
    def add_ellipse(self, H, name, pts=None):
        self.reco.append(ellipse(None, None, name, H, pts, None))

    def add_hyperbolic(self, fx, name, pts=None, domain=(-12, 12)):  # Wider default domain
        self.reco.append(hyperbolic(None, None, name, fx, domain, pts))

    def add_line(self, fx, name, pts=None, domain=(-10, 10)):
        self.reco.append(hyperbolic(None, None, name, fx, domain, pts))
    
    def _assign_styles(self):
        if self.cols_iter is None: self._reset_iterators()
        for reco in self.reco:
            if reco.col is None: reco.col = next(self.cols_iter)
            if reco.line is None: reco.line = next(self.style_iter)
            if reco.mrk_pts is None and reco.sol_pts is not None:
                reco.mrk_pts = next(self.pts_iter)
    
    def _get_combined_bounds(self):
        all_bounds = []
        if self.truth.bounds is not None: all_bounds.append(self.truth.bounds)
        for obj in self.reco:
            if isinstance(obj, ellipse) and obj.bounds is not None:
                all_bounds.append(obj.bounds)
        if not all_bounds: return None
        
        combined = {
            'x': (min(b['x'][0] for b in all_bounds), max(b['x'][1] for b in all_bounds)),
            'y': (min(b['y'][0] for b in all_bounds), max(b['y'][1] for b in all_bounds)),
            'z': (min(b['z'][0] for b in all_bounds), max(b['z'][1] for b in all_bounds))
        }
        
        return combined
    
    def _get_symmetric_limits(self, values, margin=0.15):
        max_abs = max(abs(min(values)), abs(max(values)))
        limit = max_abs * (1 + margin)
        if max_abs > 1e6: return -1e6, 1e6
        return -limit, limit
    
    def _compute_distances_to_truth(self):
        if self.truth.sol_pts is None: return None, None, None
        distances = []
        closest_idx = -1
        min_distance = float('inf')
        
        for i, obj in enumerate(self.reco):
            if obj.sol_pts is None: continue
            truth_array = np.array(self.truth.sol_pts).flatten()
            sol_array = np.array(obj.sol_pts).flatten()
            
            # If they're different sizes, truncate to minimum
            min_len = min(len(truth_array), len(sol_array))
            truth_array = truth_array[:min_len]
            sol_array = sol_array[:min_len]
            
            # Calculate Euclidean distance
            diff = truth_array - sol_array
            dist = sum(diff ** 2) # np.linalg.norm(diff)
            
            # Also calculate relative distance
            truth_magnitude = sum(truth_array**2) #np.linalg.norm(truth_array)
            distances.append((i, obj.label, dist, dist / truth_magnitude * 100, obj.sol_pts, diff))
            if dist >= min_distance: continue
            min_distance = dist
            closest_idx = i
        return distances, closest_idx, min_distance
    
    def _format_distance(self, dist_mev):
        return f"{dist_mev*10**-6:.3f} GeV^2"
    
    def _add_distance_inset(self, ax, distances, closest_idx, min_distance):
        if distances is None or not distances: return
        distances.sort(key=lambda x: x[2])  # Sort by absolute distance
        inset_text = "Distance to Truth:\n"
        inset_text += "=" * 30 + "\n"
        for i, (idx, label, dist, rel_dist, pt, diff) in enumerate(distances[:5]):  # Show top 5 closest
            marker = "→" if idx == closest_idx else " "
            display_label = label[:12] + "..." if len(label) > 15 else label
            inset_text += f"{marker} {display_label:15s}: {self._format_distance(dist)}\n"
            inset_text += f"   ({rel_dist:5.2f}% relative)\n"
        if len(distances) > 5: inset_text += f"... and {len(distances)-5} more\n"
        inset_text += "=" * 30 + "\n"
        inset_text += f"Min: {self._format_distance(min_distance)}"
        
        at = AnchoredText(inset_text, loc='upper left', frameon=True, prop=dict(size=8))
        at.patch.set_boxstyle("round,pad=0.3,rounding_size=0.2")
        at.patch.set_facecolor("white")
        at.patch.set_alpha(0.9)
        at.patch.set_edgecolor("gray")
        ax.add_artist(at)
        
        # Also add a visual line from truth to closest point if we're in 2D
        if len(self.truth.sol_pts) >= 2 and closest_idx >= 0:
            truth_pt = np.array(self.truth.sol_pts).flatten()
            closest_pt = np.array(self.reco[closest_idx].sol_pts).flatten()
            
            min_len = min(len(truth_pt), len(closest_pt), 2)
            if min_distance > 1e-6:  # More than 1 eV
                ax.plot([truth_pt[0], closest_pt[0]], 
                       [truth_pt[1], closest_pt[1]], 
                       color='black', linestyle=':', linewidth=1.5, alpha=0.7)
                
                mid_x = (truth_pt[0] + closest_pt[0]) / 2
                mid_y = (truth_pt[1] + closest_pt[1]) / 2
                
                # Offset the label perpendicular to the line
                dx = closest_pt[0] - truth_pt[0]
                dy = closest_pt[1] - truth_pt[1]
                norm = np.sqrt(dx*dx + dy*dy)
                perp_x = -dy / norm * 0.1  # Small offset
                perp_y = dx / norm * 0.1
                
                ax.text(mid_x + perp_x, mid_y + perp_y, 
                       f"{self._format_distance(min_distance)}",
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.2", 
                       facecolor="white", alpha=0.8,
                       edgecolor="gray"))

    def compile3D(self):
        """Compile and plot 3D visualization"""
        fig = plt.figure(figsize=(12, 8))
        self.ax = plt.axes(projection="3d")
        self._assign_styles()
        
        t_values = np.linspace(0, 2 * np.pi, self.pts)
        all_objects = self.reco + [self.truth]
        for obj in all_objects:
            if not isinstance(obj, ellipse): continue
            obj.compile(t_values)
        ellipse_bounds = self._get_combined_bounds()
        for obj in all_objects:
            if not isinstance(obj, hyperbolic): continue 
            obj.compile()
            if ellipse_bounds is not None: obj.clip_to_bounds(ellipse_bounds)
            else: obj.points = obj.raw_points
        for obj in all_objects:
            if obj.points is not None and len(obj.points) > 0:
                self.ax.plot(obj.points[:, 0], obj.points[:, 1], obj.points[:, 2],
                           color=obj.col, linestyle=obj.line, linewidth=2,
                           label=obj.label, alpha=1.0)
        
        # Then markers on top
        for obj in all_objects:
            if obj.sol_pts is not None:
                self.ax.scatter(obj.sol_pts[0], obj.sol_pts[1], obj.sol_pts[2],
                              marker=obj.mrk_pts, s=80, color=obj.col,
                              edgecolor='black', linewidth=1, label=f"{obj.label} (solution)")
        
        # Set axis labels
        self.ax.set_xlabel("x (MeV)")
        self.ax.set_ylabel("y (MeV)")
        self.ax.set_zlabel("z (MeV)")
        
        self.ax.grid(True, alpha=0.3, linestyle=':')
        self._set_3d_axis_limits_symmetric(self.ax, all_objects)
        distances, closest_idx, min_distance = self._compute_distances_to_truth()
        if distances:
            truth_pt = np.array(self.truth.sol_pts).flatten()
            closest_pt = np.array(self.reco[closest_idx].sol_pts).flatten()
            self.ax.plot([truth_pt[0], closest_pt[0]], 
                        [truth_pt[1], closest_pt[1]], 
                        [truth_pt[2], closest_pt[2]],
                        color='black', linestyle=':', linewidth=1.5, alpha=0.7)
            
            # Add text annotation
            self.ax.text2D(0.02, 0.98, 
                          f"Min distance: {self._format_distance(min_distance)}\nClosest: {self.reco[closest_idx].label}",
                          transform=self.ax.transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor="white", alpha=0.9,
                                   edgecolor="gray"))
        
        handles, labels = self.ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]][:5]
        if unique: self.ax.legend(*zip(*unique), loc='upper right', fontsize=9)
            
        plt.tight_layout()
        plt.show()
    
    def _set_3d_axis_limits_symmetric(self, ax, objects):
        """Set 3D axis limits symmetric around (0,0,0)"""
        x_vals, y_vals, z_vals = [], [], []
        
        for obj in objects:
            if obj.points is not None and len(obj.points) > 0:
                x_vals.extend(obj.points[:, 0])
                y_vals.extend(obj.points[:, 1])
                z_vals.extend(obj.points[:, 2])
            
            if obj.sol_pts is not None:
                # Take first 3 coordinates only
                pt = np.array(obj.sol_pts).flatten()[:3]
                if len(pt) > 0: x_vals.append(pt[0])
                if len(pt) > 1: y_vals.append(pt[1])
                if len(pt) > 2: z_vals.append(pt[2])
        
        # Set symmetric limits for each axis
        margin = 0.15
        
        if x_vals:
            x_lim = self._get_symmetric_limits(x_vals, margin)
            ax.set_xlim3d(x_lim[0], x_lim[1])
        if y_vals:
            y_lim = self._get_symmetric_limits(y_vals, margin)
            ax.set_ylim3d(y_lim[0], y_lim[1])
        if z_vals:
            z_lim = self._get_symmetric_limits(z_vals, margin)
            ax.set_zlim3d(z_lim[0], z_lim[1])

    def compile2D(self):
        fig = plt.figure(figsize=(10, 8))
        self.ax = plt.axes()
        self.make_axis(self.ax, ("Sx", "Sy"))
      
        self._assign_styles()
        
        t_values = np.linspace(0, 2 * np.pi, self.pts)
        all_objects = self.reco + [self.truth]
        
        for obj in all_objects:
            if not isinstance(obj, ellipse): continue
            obj.compile(t_values)
        
        ellipse_bounds = self._get_combined_bounds()
        
        for obj in all_objects:
            if not isinstance(obj, hyperbolic): continue
            obj.compile()
            if ellipse_bounds is not None: obj.clip_to_bounds(ellipse_bounds)
            else: obj.points = obj.raw_points
        
        self._set_axis_limits_symmetric(self.ax, all_objects, ("x", "y"))
        
        for obj in all_objects:
            if obj.points is not None and len(obj.points) > 0:
                self.ax.plot(obj.points[:, 0], obj.points[:, 1],
                           color=obj.col, linestyle=obj.line,
                           linewidth=3, label=obj.label, alpha=1.0)
            
            if obj.sol_pts is not None:
                pt = np.array(obj.sol_pts).flatten()
                if len(pt) >= 2:
                    self.ax.plot(pt[0], pt[1],
                               marker=obj.mrk_pts, markersize=10,
                               color=obj.col, linestyle='',
                               markeredgecolor='black', markeredgewidth=1,
                               label=f"{obj.label} (solution)")
        
        # Compute distances and add inset
        distances, closest_idx, min_distance = self._compute_distances_to_truth()
        if distances: self._add_distance_inset(self.ax, distances, closest_idx, min_distance)
        
        # Handle legend
        handles, labels = self.ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]][:6]
        if unique: self.ax.legend(*zip(*unique), loc='upper right', fontsize=10)
        plt.tight_layout()
        plt.show()
    
    def _set_axis_limits_symmetric(self, ax, objects, dims=("x", "y")):
        """Set axis limits symmetric around (0,0)"""
        x_idx = 0 if dims[0] == 'x' else 1 if dims[0] == 'y' else 2
        y_idx = 0 if dims[1] == 'x' else 1 if dims[1] == 'y' else 2
        
        # Collect all visible points
        x_vals, y_vals = [], []
        
        for obj in objects:
            if obj.points is not None and len(obj.points) > 0:
                x_vals.extend(obj.points[:, x_idx])
                y_vals.extend(obj.points[:, y_idx])
            
            if obj.sol_pts is not None:
                pt = np.array(obj.sol_pts).flatten()
                if len(pt) > x_idx: x_vals.append(pt[x_idx])
                if len(pt) > y_idx: y_vals.append(pt[y_idx])
        
        # Set symmetric limits for each axis
        margin = 0.15
        
        if x_vals:
            x_lim = self._get_symmetric_limits(x_vals, margin)
            ax.set_xlim(x_lim[0], x_lim[1])
        if y_vals:
            y_lim = self._get_symmetric_limits(y_vals, margin)
            ax.set_ylim(y_lim[0], y_lim[1])

    def compile2D_Proj(self, save=None, show_inset=True):
        self.fig, self.ax = plt.subplots(1, 3, figsize=(20, 8))
        
        # Assign styles once
        self._assign_styles()
        
        # Set up axes
        projections = [("x", "y"), ("x", "z"), ("y", "z")]
        
        t_values = np.linspace(0, 2 * np.pi, self.pts)
        all_objects = self.reco + [self.truth]
        
        # First compile ellipses to get bounds
        for obj in all_objects:
            if isinstance(obj, ellipse):
                obj.compile(t_values)
        
        ellipse_bounds = self._get_combined_bounds()
        for obj in all_objects:
            if not isinstance(obj, hyperbolic): continue
            obj.compile()
            if ellipse_bounds is not None: obj.clip_to_bounds(ellipse_bounds)
            else: obj.points = obj.raw_points
        
        distances, closest_idx, min_distance = self._compute_distances_to_truth()
        for idx, (dims, ax) in enumerate(zip(projections, self.ax)):
            self.make_axis(ax, dims)
            self._set_axis_limits_symmetric(ax, all_objects, dims)
            x_idx = 0 if dims[0] == 'x' else 1 if dims[0] == 'y' else 2
            y_idx = 0 if dims[1] == 'x' else 1 if dims[1] == 'y' else 2
            for obj in all_objects:
                if obj.points is not None and len(obj.points) > 0:
                    ax.plot(obj.points[:, x_idx], obj.points[:, y_idx],
                           color=obj.col, linestyle=obj.line, linewidth=3,
                           alpha=0.8, label=obj.label if idx == 0 else "")
            
            for obj in all_objects:
                if obj.sol_pts is not None:
                    pt = np.array(obj.sol_pts).flatten()
                    if len(pt) > max(x_idx, y_idx):
                        ax.scatter(pt[x_idx], pt[y_idx],
                                  marker=obj.mrk_pts, s=60, color=obj.col,
                                  edgecolor='black', linewidth=3, zorder=5,
                                  label=f"{obj.label} (sol)" if idx == 0 else "")
            
            # Add distance inset only to first subplot
            if idx == 0 and show_inset and distances:
                self._add_distance_inset(ax, distances, closest_idx, min_distance)
                
                # Also add a line from truth to closest in the first projection
                if closest_idx >= 0 and len(self.truth.sol_pts) >= 2:
                    truth_pt = np.array(self.truth.sol_pts).flatten()
                    closest_pt = np.array(self.reco[closest_idx].sol_pts).flatten()
                    
                    # Draw a dashed line from truth to closest solution
                    ax.plot([truth_pt[0], closest_pt[0]], 
                           [truth_pt[1], closest_pt[1]], 
                           color='black', linestyle=':', linewidth=1.5, alpha=0.7)
                
                # Legend only on first subplot
                handles, labels = ax.get_legend_handles_labels()
                unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]][:6]
                if unique: ax.legend(*zip(*unique), loc='upper right', fontsize=9)
        
        plt.tight_layout(pad=2.0)
        if save is not None: plt.savefig(str(save) + ".pdf", dpi=300, bbox_inches='tight')
        else: plt.show()
        plt.close()

    def make_axis(self, ax, dims=("x", "y")):
        """Setup axis labels and style"""
        i, j = dims
        ax.set_xlabel(f"{i} (MeV)")
        ax.set_ylabel(f"{j} (MeV)")
        ax.grid(True, alpha=0.7, linestyle=':')
        
        # Add (0,0) crosshairs for visual reference
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Keep equal aspect ratio
        ax.axis('equal')
