from atomics import *
from visualize import *

class data:
    def __init__(self):
        self.npts   = 100
        self.matrix = None
        self.normal = None
        self.center = None
        self.r0     = None



class Ellipsoid(cosmetic):
    def __init__(self, obj, ax):
        cosmetic.__init__(self, ax)
        self.show_eigen = True
        self.data = None
        self.el   = obj

    def equation(self, n_points):
        u = self._lin(0, 2*np.pi, n_points)
        v = self._lin(0,   np.pi, n_points)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        return np.stack([i.flatten() for i in [x, y, z]])
 
    # (x - p)^T * q3 * (x - p) + L^T * p + c = 0
    def make(self, n_points = 10):
        pnt = self.equation(n_points)

        q3 = self.data.matrix[:3, :3]
        L  = self.data.matrix[:3,  3].reshape(3, 1)
        c  = self.data.matrix[3, 3]
        p  = - np.linalg.inv(q3).dot(L) 
        k  = - (L.T.dot(p) + c)
        
        vl, vx = np.linalg.eigh(q3/k)
        r = np.sqrt(1 / vl)
      
        scl = vx.dot( np.diag(r).dot(pnt) ) + p
        return [scl[i, :].reshape((n_points, n_points)) for i in range(3)], vx, p

    def plot(self):
        (x, y, z), vx, p = self.make(10)
        self.max_x, self.min_x = np.max(x), np.min(x)
        self.max_y, self.min_y = np.max(y), np.min(y)
        self.max_z, self.min_z = np.max(z), np.min(z)
        self._surface(x, y, z)
#        if not self.show_eigen: return
#        _x, _y, _z = p.flatten().tolist()
#        vx = vx.T
#        for i in range(3):
#            v = vx[i].tolist()
#            self._quiver(_x, _y, _z, v[0], v[1], v[2])


class Ellipse(cosmetic):

    def __init__(self, obj, ax):
        cosmetic.__init__(self, ax)
        self.data = None
        self.el = obj

    def equation(self, a, b, phi):
        return self.data.matrix.dot([np.cos(phi), np.sin(phi), np.ones_like(phi)])

    def make(self, n_points = 100):
        vl, vx = np.linalg.eigh(self.data.matrix)
      
        idx = vl.argsort()[::-1]
        vl, vx = vl[idx], vx[:,idx]

        r = np.sqrt(vl)
        v1, v2 = vx[:,0], vx[:,1]
        a, b   = v1 * r[0], v2 * r[1]
        return self.equation(a, b, self._lin(0, 2*np.pi, n_points))

    def plot(self):
        x, y, z = self.make(self.data.npts)
        self.max_x, self.min_x = np.max(x), np.min(x)
        self.max_y, self.min_y = np.max(y), np.min(y)
        self.max_z, self.min_z = np.max(z), np.min(z)

        self._plot3(x, y, z)



class Plane(cosmetic):

    def __init__(self, obj, ax):
        cosmetic.__init__(self, ax)
        self.data = None
        self.xmin, self.ymin = -1, -1
        self.xmax, self.ymax =  1,  1
        self.show_normal = True
        self.pl = obj

    def equation(self, r0, r1, r2, u, v):
        return r0 + r1 * u + r2 * v

    def make(self, n_points = 100):
        n = self.data.normal / np.linalg.norm(self.data.normal)
        if abs(n[0]) > 1e-6 or abs(n[1]) > 1e-6: perp = np.array([-n[1], n[0], 1])
        else: perp = np.array([1, 0, 0])
        p1 = perp / np.linalg.norm(perp)
        p2 = np.cross(n, p1)
        p2 = p2 / np.linalg.norm(p2)
        u, v = np.meshgrid(
                self._lin(self.xmin, self.xmax, n_points), 
                self._lin(self.ymin, self.ymax, n_points)
        )
        x = self.equation(self.data.center[0], p1[0], p2[0], u, v)
        y = self.equation(self.data.center[1], p1[1], p2[1], u, v)
        z = self.equation(self.data.center[2], p1[2], p2[2], u, v)
        return x, y, z, n
    
    def plot(self):
        x, y, z, n = self.make(self.data.npts)
        self.max_x, self.min_x = np.max(x), np.min(x)
        self.max_y, self.min_y = np.max(y), np.min(y)
        self.max_z, self.min_z = np.max(z), np.min(z)

        self._surface(x, y, z)
        if not self.show_normal: return 
        self._quiver(
            self.data.center[0], 
            self.data.center[1], 
            self.data.center[2],
            n[0], n[1], n[2]
        )

        self._scatter3(
            self.data.center[0], 
            self.data.center[1], 
            self.data.center[2]
        )

    
class Line(cosmetic):

    def __init__(self, obj, ax):
        cosmetic.__init__(self, ax)
        self.data = None
        self.xmin, self.ymin = -1, -1
        self.xmax, self.ymax =  1,  1
        self.pl = obj 

    def equation(self, x, y):
        pass

    def make(self, n_points = 100):
        pass

    def plot(self):
        pass




