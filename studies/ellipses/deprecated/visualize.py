from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

class cosmetic:

    def __init__(self, ax = None):
        self.ax = ax
        self.color = "b-"
        self.label = "untitled"
        self.alpha = 0.8
        self.linewidth = 2
        self.font_size = 12
        self.marker_size = 2
        self.marker_sym  = "o"

        self.max_x, self.min_x = None, None
        self.max_y, self.min_y = None, None
        self.max_z, self.min_z = None, None

    def _lin(self, min_, max_, pts):
        return np.linspace(min_, max_, pts)

    def _plot3(self, x, y, z):
        self.ax.plot(
            x, y, z, 
            self.color, 
            label = self.label, 
            linewidth = self.linewidth,
            alpha = self.alpha
        )

    def _scatter3(self, x, y, z):
        self.ax.scatter(
                x, y, z, 
                s = self.marker_size, 
                c = self.color, 
                marker = self.marker_sym, 
                edgecolor = "k", 
                label = f'{self.label}'
        )

    def _surface(self, x, y, z):
        self.ax.plot_surface(
                x, y, z, 
                color = self.color, 
                alpha = self.alpha,
                antialiased = False
        )


    def _wire(self, x, y, z):
        self.ax.plot_wireframe(
                x, y, z, 
                color = self.color, 
                alpha = self.alpha,
                antialiased = False
        )

 

    def _quiver(self, x, y, z, nx, ny, nz):
        def lx(nx, ny, nz): return (nx**2 + ny**2 + nz**2)**0.5
        self.ax.quiver3D(
                x, y, z, 
                nx, ny, nz, 
                length = lx(nx, ny, nz), 
                color = self.color,
                pivot = "tail",
                linewidth = self.linewidth
        )
 

    def plot(self): return None


class figure(cosmetic):
    def __init__(self, fig_size = (14, 10), prj = "3d"):
        cosmetic.__init__(self)
        self.fig = plt.figure(figsize = fig_size)
        self.ax = self.fig.add_subplot(111, projection = prj)
        self.auto_lims = True
        self.objects = {}

        self.max_x, self.min_x = None, None
        self.max_y, self.min_y = None, None
        self.max_z, self.min_z = None, None


    def get_min(self, v, x):
        if v is None and x is None: return None
        if v is None and x is not None: return x
        if v is not None and x is None: return v
        if v > x: return x
        return v

    def get_max(self, v, x):
        if v is None and x is None: return None
        if v is None and x is not None: return x
        if v is not None and x is None: return v
        if v < x: return x
        return v

    def scan_lims(self):
        for i in sum(self.objects.values(), []):
            if not self.auto_lims: break
            self.max_x = self.get_max(self.max_x, i.max_x)
            self.max_y = self.get_max(self.max_y, i.max_y)
            self.max_z = self.get_max(self.max_z, i.max_z)

            self.min_x = self.get_min(self.min_x, i.min_x)
            self.min_y = self.get_min(self.min_y, i.min_y)
            self.min_z = self.get_min(self.min_z, i.min_z)

        self.axis_lims("x", self.min_x, self.max_x)
        self.axis_lims("y", self.min_y, self.max_y)
        self.axis_lims("z", self.min_z, self.max_z)


    def add_object(self, name, obj):
        if name in self.objects: return None 
        self.objects[name] = obj.plots

    def axis_label(self, dim, name):
        if dim == "x": self.ax.set_xlabel(name, fontsize = self.font_size)
        if dim == "y": self.ax.set_ylabel(name, fontsize = self.font_size)
        if dim == "z": self.ax.set_zlabel(name, fontsize = self.font_size)

    def axis_lims(self, dim, min_, max_):
        if dim == "x": self.ax.set_xlim(min_, max_)
        if dim == "y": self.ax.set_ylim(min_, max_)
        if dim == "z": self.ax.set_zlim(min_, max_)

    def plot_title(self, name, font_size): 
        self.ax.set_title(name, fontsize = font_size)

    def show(self):
        lx = sum(list(self.objects.values()), [])
        if not len(lx): return 
        for i in lx: i.plot()
        self.scan_lims()
        plt.tight_layout()
        plt.show()


