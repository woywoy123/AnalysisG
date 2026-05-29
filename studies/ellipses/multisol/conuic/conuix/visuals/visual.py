import matplotlib.pyplot as plt
import numpy as np
import itertools

import pyvista as pv
import pyvistaqt as pvqt
pv.global_theme.allow_empty_mesh = True

class cfg_t:

    def __init__(self, col, line, label, s_pts, mrk):
        self.col     = col
        self.line    = line
        self.label   = label
        self.sol_pts = s_pts
        self.marker  = mrk

        self.points = None

    def set_scatter(self, ax, idx = None, idy = None, idz = None):
        if self.sol_pts is None: return
        dm = sum([i is not None for i in [idx, idy, idz]])
        dic  = {"marker" : self.marker, "color" : self.col}
        dic |= {"edgecolor": "black", "linewidth" : 2, "label" : self.label}
        if idx is None or idy is None: idx, idy = 0, 1
        if dm == 3: ax.scatter(self.sol_pts[idx], self.sol_pts[idy], self.sol_pts[idz], **dic)
        if dm == 2: ax.scatter(self.sol_pts[idx], self.sol_pts[idy], **dic)

    def set_points(self, ax, idx = None, idy = None, idz = None):
        if self.points is None: return
        dm = sum([i is not None for i in [idx, idy, idz]])
        dic  = {"color" : self.col, "linestyle" : self.line}
        dic |= {"linewidth" : 1, "label" : self.label, "alpha" : 0.8}
        if idx is None or idy is None: idx, idy = 0, 1
        if dm == 3: ax.plot(self.points[:, idx], self.points[:, idy], self.points[:, idz], **dic)
        if dm == 2: ax.plot(self.points[:, idx], self.points[:, idy], **dic)

    def _x(self, i, t, pts): return np.linspace(i, t, pts)
    def _y(self, i, t, pts): return np.linspace(i, t, pts)
    def _z(self, i, t, pts): return np.linspace(i, t, pts)


class cfx_t:
    
    def __init__(self, name, color, scale, mn = 1.0, mx = 1.0):
        self.scale = scale
        self.min  = -mn * scale
        self.max  =  mx * scale
        self.pts  = 100
        self.name = name
        self.col  = color

    def _s(self): return np.linspace(self.min, self.max, self.pts)

    def sxT(self, x, y, z): return x
    def syT(self, x, y, z): return y
    def szT(self, x, y, z): return z

    @property
    def x(self): return self._s()
    
    @property
    def y(self): return self._s()
 
    @property
    def z(self): return self._s()
    
    def fx(self, x, y, z): return x + y + z

    @property
    def get(self): 
        x, y, z = np.meshgrid(self.x, self.y, self.z, indexing = "ij")
        x, y, z = self.sxT(x, y, z), self.syT(x, y, z), self.szT(x, y, z)
        gr = pv.StructuredGrid(x, y, z)
        gr.point_data["F"] = self.fx(x, y, z).ravel(order = "F")
        return gr.contour(isosurfaces = [0], scalars = "F")

    @property
    def inits(self): return str(self.__class__.__name__.__str__())

    def add_msh(self, pltr):
        if "Two" not in self.inits and "Particle" not in self.inits: return False
        smooth_w_taubin = self.get#.smooth_taubin(n_iter=50, pass_band=0.05)
        op  = {"color" : self.col, "smooth_shading": True, "label" : self.name}
        op |= {"show_edges": True, "line_width" : 0.01, "specular" : 0.1}
        op |= {"opacity" : 0.8, "show_scalar_bar" : False}
        pltr.add_mesh(smooth_w_taubin, **op)
        return True

    def add_pts(self, pltr):
        if self.inits != "Point": return False
        try: smooth_w_taubin = self.get#.smooth_taubin(n_iter=50, pass_band=0.05)
        except TypeError: return False
        op  = {"color" : self.col, "smooth_shading": True, "label" : self.name}
        op |= {"show_edges": False, "line_width" : 0.0001, "specular" : 0.001}
        op |= {"opacity" : 0.6, "show_scalar_bar" : False}
        try: pltr.add_mesh(smooth_w_taubin, **op)
        except TypeError: return False
        return True

    def add_ellipse(self, pltr):
        if "Ellipse" not in self.inits: return False
        phi = np.linspace(0, 2*np.pi, 200)
        circ = np.vstack([np.cos(phi), np.sin(phi), np.ones_like(phi)])
        pts = np.column_stack(self.Q @ circ)
        disk = pv.PolyData(pts).delaunay_2d()
        slab = disk.extrude((0, 0, 5000), capping=True)

        op  = {"color" : self.col, "smooth_shading": True, "label" : self.name}
        op |= {"show_edges": False, "line_width" : 0.0, "specular" : 0.1}
        op |= {"opacity" : 0.8, "show_scalar_bar" : False}
        try: pltr.add_mesh(slab, **op)
        except TypeError: return False
        return True


def PlotQuadrics(obj):
    p = pv.Plotter(window_size = (1890, 900)) 
#    pv.Axes().origin
    for i in obj:
        if i.add_msh(p): continue
        if i.add_pts(p): continue
        if i.add_ellipse(p): continue

    p.show_grid(font_size=10,color='#3a5f7a',xtitle="Sx (MeV)", ytitle="Sy (MeV)", ztitle="Sz (MeV)")
    p.set_background('#0c141f')
    p.add_legend()
    p.export_vtksz("surf.vtksz")
    p.show()



class packet:
    def __init__(self, pts=100, domain = [[-1000, 1000], [-1000, 1000], [-1000, 1000]], scale = 1):
        self.axes = [[k * scale for k in i] for i in domain]
        self.shapes = []
        self.cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        self.style = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
        self.pts_ = ["o", "s", "^", "v", "D", "P", "X"]
        
        # Create iterators
        self.cols_iter = None
        self.style_iter = None
        self.pts_iter = None
        self.num_pts = pts
   
    def add_truth(self, H, name = 'Truth Neutrino', pts = None, line = "solid"):
        self.shapes.append(ellipse_t("red", line, name, H, pts, "*"))

    def add_ellipse(self, H, name, pts=None, line = None, col = None):
        self.shapes.append(ellipse_t(col, line, name, H, pts, None))

    def add_line(self, fx, name, pts=None, domain=[[-100, 100]], line = None, col = None, mrk = None):
        self.shapes.append(line_t(col, line, name, fx, domain, pts, mrk))

    def _assign_styles(self):
        if self.cols_iter is None: self._reset_iterators()
        for s in self.shapes:
            if s.col is None:    s.col    = next(self.cols_iter)
            if s.line is None:   s.line   = next(self.style_iter)
            if s.marker is None: s.marker = next(self.pts_iter)
    
    def compile3D(self, label = ["Sx (MeV)", "Sy (MeV)", "Sz (MeV)"]):
#        plt.style.use('dark_background')
        fig = plt.figure(figsize=(12, 12))
        self.ax = plt.axes(projection="3d")
        self._assign_styles()
        for s in self.shapes: s.compile(self.num_pts)
        for s in self.shapes: s.set_points(self.ax, 0, 1, 2)
        for s in self.shapes: s.set_scatter(self.ax, 0, 1, 2)
        self.ax.set_xlabel(label[0])
        self.ax.set_ylabel(label[1])
        self.ax.set_zlabel(label[2])
        self.ax.grid(True, alpha=0.3, linestyle=':')
        handles, labels = self.ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        if unique: self.ax.legend(*zip(*unique), loc='upper right', fontsize=9)
        plt.tight_layout()
        plt.show()
    
    def compile2D(self, label = ["Sx (MeV)", "Sy (MeV)", "Sz (MeV)"]):
#        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 8))

        self.ax = plt.axes()
        self._assign_styles()
        self._make_axis()

        for s in self.shapes: s.compile(self.num_pts)
        for s in self.shapes: s.set_points(self.ax, 0, 1)
        for s in self.shapes: s.set_scatter(self.ax, 0, 1)

        handles, labels = self.ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        if unique: self.ax.legend(*zip(*unique), loc='upper right', fontsize=10)
        self.ax.set_xlabel(label[0])
        self.ax.set_ylabel(label[1])
        plt.tight_layout()
        plt.show()
    
    def compile2DProj(self, label = ["Sx (MeV)", "Sy (MeV)", "Sz (MeV)"]):
#        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(1, 3, figsize=(20, 8))
        self._assign_styles()
        self._make_axis()
        projs = [(label[0], label[1]), (label[0], label[2]), (label[1], label[2])]
        for s in self.shapes: s.compile(self.num_pts)
        for idx, (dims, ax) in enumerate(zip(projs, self.ax)):
            x_idx = 0 if dims[0] == label[0] else 1 if dims[0] == label[1] else 2
            y_idx = 0 if dims[1] == label[0] else 1 if dims[1] == label[1] else 2
            for s in self.shapes: s.set_points(ax , x_idx, y_idx)
            for s in self.shapes: s.set_scatter(ax, x_idx, y_idx)
            ax.set_xlabel(projs[idx][0]); ax.set_ylabel(projs[idx][1])

        plt.tight_layout(pad=2.0)
        plt.show(block = False)
        plt.pause(0.4)
        plt.close()

    def _make_axis(self):
        try: 
            ax = self.ax
            for i in range(len(ax)):
                self.ax = ax[i]
                self._make_axis()
            self.ax = ax
            return 
        except: pass

        self.ax.grid(True, alpha=0.7, linestyle=':')
        self.ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        self.ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        for i in range(len(self.axes)):
            mi, mx = self.axes[i]
            try:
                if i == 0: self.ax.set_xlim3d(mi, mx)
                if i == 1: self.ax.set_ylim3d(mi, mx)
                if i == 2: self.ax.set_zlim3d(mi, mx)
            except AttributeError:
                if i == 0: self.ax.set_xlim(mi, mx)
                if i == 1: self.ax.set_ylim(mi, mx)
        self.ax.axis('equal')       

    def _reset_iterators(self):
        self.cols_iter = itertools.cycle(self.cols)
        self.style_iter = itertools.cycle(self.style)
        self.pts_iter = itertools.cycle(self.pts_)
 


class line_t(cfg_t):
    def __init__(self, col = None, line = None, label = None, fx = None, domain = None, sols = None, mrk = None):
        cfg_t.__init__(self, col, line, label, sols, mrk)
        self.dim = len(domain)
        self.domain = domain[0] if self.dim == 1 else domain
        self.fx = fx

    def compile(self, pts = 100):
        if self.fx is None: return
        if self.dim == 1: 
            s = np.linspace(self.domain[0], self.domain[1], pts)
            self.points = np.array([self.fx(t) for t in s])
        if self.dim == 2:
            x = self._x(self.domain[0][0], self.domain[0][1], pts)
            y = self._y(self.domain[1][0], self.domain[1][1], pts)
            x, y = np.meshgrid(x, y) #, indexing = "ij")
            x, y = x.flatten(), y.flatten()
            self.points = np.array([self.fx(t, j) for t, j in zip(x,y)])
            self.points = self.points.reshape((-1, 3))
            print(self.points)


class ellipse_t(cfg_t):

    def __init__(self, col=None, line=None, label=None, H_tilde=None, sols=None, mrk=None):
        cfg_t.__init__(self, col, line, label, sols, mrk)
        self.H = H_tilde

    def compile(self, pts = None):
        if self.H is None: return
        t_values = np.linspace(0, 2 * np.pi, 100)
        self.points = np.array([self.H.dot(np.array([np.cos(t), np.sin(t), 1])) for t in t_values])


