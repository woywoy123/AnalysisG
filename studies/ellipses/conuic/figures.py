import matplotlib.pyplot as plt
import numpy as np
import math

class ellipse:
    def __init__(self, col = None, line = None, label = None, H_tilde = None, sols = None, mrk = None):
        self.col     = col
        self.line    = line
        self.H       = H_tilde
        self.label   = label
        self.sol_pts = sols
        self.points  = None
        self.mrk_pts = mrk

    def compile(self, t_values):
        if self.H is None: return 
        self.points = np.array([self.H.dot(np.array([np.cos(t), np.sin(t), 1])) for t in t_values])

class hyperbolic:

    def __init__(self, col = None, line = None, label = None, fx = None, domain = None, pts = None):
        self.col   = col
        self.label = label 
        self.line  = line
        self.fx    = fx
        self.H     = None
        self.points = np.linspace(domain[0], domain[1], 1000)

        self.sol_pts = pts
        self.mrk_pts = None

    def compile(self, t_values):
        if self.fx is None: return 
        self.points = np.array([self.fx(i) for i in self.points])

class packet:

    def __init__(self, truth = None, tru_pts = None, pts = 1000, inst = None):
        self.truth = ellipse("red", "solid", 'Truth Neutrino', truth, tru_pts, "*")
        self.reco  = []
        self.pts   = pts
        self.instance = inst
        self.cols  = ["black", "orange", "green", "blue", "black", "orange", "green",  "blue", "orange", "green", "blue"]
        self.style = ["-."   ,     "-.",    "-.",   "-.", ":"    ,     ":",    ":",       ":",    "--",    "--",    "--"]
        self.pts_  = ["x"    ,      "x",     "x",    "x", "o"    ,     "o",    "o",       "+",    "+",    "+",       "+"]
 
    def add_ellipse(self, H, name, pts = None, enable = False):
        if pts is not None and enable: pts = self.instance.rz.T.dot(pts)
        if H   is not None and enable: H   = self.instance.rz.T.dot(H)
        self.reco.append(ellipse(None, None, name, H, pts, None))

    def add_hyperbolic(self, fx, name, pts = None, domain = (-1, 1)):
        self.reco.append(hyperbolic(None, None, name, fx, domain, pts))

    def add_line(self, fx, name, pts = None, domain = (-1, 1)):
        self.reco.append(hyperbolic(None, None, name, fx, domain, pts))

    def compile3D(self):
        fig = plt.figure(figsize = (15, 10))
        self.ax = plt.axes(projection = "3d")
        self.make_axis(self.ax, ("x", "y"))
        self.make_axis(self.ax, ("y", "z"))
       
        for i in range(len(self.reco)): 
            self.reco[i].col  = next(self.cols)
            self.reco[i].line = next(self.style)
            if self.reco[i].sol_pts is None: continue
            self.reco[i].mrk_pts = next(self.pts_)

        t_values = np.linspace(0, 2*np.pi, self.pts)
        self.reco += [self.truth]
        for i in self.reco: i.compile(t_values)
        for i in self.reco:
            if i.sol_pts is not None: 
                self.ax.scatter(i.sol_pts[0], i.sol_pts[1], i.sol_pts[2], marker = i.mrk_pts, label = i.label, color = i.col)
            if i.points  is not None: 
                self.ax.plot(i.points[:,0], i.points[:,1], i.points[:,2], color = i.col, linestyle = i.line, linewidth=2)

        self.ax.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def compile2D(self):
        fig = plt.figure(figsize = (15, 10))
        self.ax = plt.axes()
        self.make_axis(self.ax, ("x", "y"))
      
        for i in range(len(self.reco)): 
            self.reco[i].col  = next(self.cols)
            self.reco[i].line = next(self.style)
            if self.reco[i].sol_pts is None: continue
            self.reco[i].mrk_pts = next(self.pts_)

        t_values = np.linspace(0, 2*np.pi, self.pts)
        self.reco += [self.truth]
        for i in self.reco: i.compile(t_values)
        for i in self.reco:
            if i.points  is not None: 
                self.ax.plot(i.points[:, 0], i.points[:, 1], color=i.col, linestyle = i.line, linewidth=1)
            if i.sol_pts is not None: 
                self.ax.plot(i.sol_pts[0], i.sol_pts[1], marker = i.mrk_pts, markersize=10, label = i.label)
        self.ax.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def compile2D_Proj(self, save = None):
        self.fig, self.ax = plt.subplots(1, 3, figsize = (15, 10))
        self.make_axis(self.ax[0], ("x", "y"))
        self.make_axis(self.ax[1], ("x", "z"))
        self.make_axis(self.ax[2], ("y", "z"))
        
        for i in range(len(self.reco)): 
            self.reco[i].col  = next(self.cols)
            self.reco[i].line = next(self.style)
            self.reco[i].label = None
            if self.reco[i].sol_pts is None: continue
            self.reco[i].mrk_pts = next(self.pts_)

        t_values = np.linspace(0, 2*np.pi, self.pts)
        self.reco += [self.truth]
        for i in self.reco: i.compile(t_values)
        for i in self.reco:
            if i.points is not None:
                self.ax[0].plot(i.points[:, 0], i.points[:, 1], color=i.col, linestyle = i.line, linewidth=1) #, zorder = 1)
                self.ax[1].plot(i.points[:, 0], i.points[:, 2], color=i.col, linestyle = i.line, linewidth=1) #, zorder = 1)
                self.ax[2].plot(i.points[:, 1], i.points[:, 2], color=i.col, linestyle = i.line, linewidth=1) #, zorder = 1)

            if i.sol_pts is not None and i.label is not None: 
                self.ax[0].scatter(i.sol_pts[0], i.sol_pts[1], label = i.label, marker = i.mrk_pts)
                self.ax[1].scatter(i.sol_pts[0], i.sol_pts[2], label = i.label, marker = i.mrk_pts)
                self.ax[2].scatter(i.sol_pts[1], i.sol_pts[2], label = i.label, marker = i.mrk_pts)

        self.ax[0].legend(loc='best')
        plt.tight_layout()
        if save is not None: plt.savefig(str(save))
        else: plt.show()
        plt.close()

    def make_axis(self, ax, dims = ("x", "y")):
        try:
            while len(self.reco) >= len(self.style):
                self.cols  += self.cols   
                self.style += self.style 
                self.pts_  += self.pts_  
       
            self.cols  = iter(self.cols)           
            self.style = iter(self.style)
            self.pts_  = iter(self.pts_)
        except: pass


        i, j = dims
        ax.set_xlabel(i + "1 (GeV)") 
        ax.set_ylabel(j + "1 (GeV)")
        ax.grid(True, alpha=0.3)
        ax.autoscale(axis = i, enable = True)
        ax.autoscale(axis = j, enable = True)
        #ax.axis('equal')
