from atomics import *

def h_tilde(obj):
    htc = np.array([
        [1       , 0    , 0], 
        [obj.tpsi, 0    , 0], 
        [0       , obj.o, 0]
    ]) * (1.0 / obj.o)

    ht1 = np.array([
        [0, 0,        -1], 
        [0, 0, -obj.tpsi], 
        [0, 0,         0]
    ]) * (obj.lep.b * obj.cpsi /obj.o)

    ht2 = np.array([
        [0, 0, -obj.tpsi],
        [0, 0,         1],
        [0, 0,         0]
    ]) * obj.cpsi
    return htc, ht1, ht2

def A_mu(obj):
    atc = nulls(4, 4)
    atc[0][0] = (obj.lep.mass/obj.lep.e)**2
    atc[1][1] = 1
    atc[2][2] = 1
    atc[3][3] = - obj.lep.mass**2
    
    at1 = nulls(4, 4)
    at1[0][3] = obj.lep.b2
    at1[3][0] = obj.lep.b2
    at1[3][3] = - 2 * obj.lep.b * obj.lep.e

    at2 = nulls(4, 4)
    at2[3][3] = - obj.lep.b2

    return np.array(atc), np.array(at1), np.array(at2)

def A_b(obj):
    c, s = obj.cos, obj.sin

    # constants
    atc = nulls(4, 4)
    atc[0][0] = 1 - obj.jet.b2 * (c ** 2)
    atc[0][1] =   - obj.jet.b2 * c * s
    atc[1][0] =   - obj.jet.b2 * c * s
    atc[1][1] = 1 - obj.jet.b2 * (s ** 2)

    atc[2][2] = 1
    atc[3][3] = - obj.lep.mass2

    # multiply by (Sx cos + Sy sin)
    at1 = nulls(4, 4)
    at1[0][3] = obj.jet.b2 * c
    at1[1][3] = obj.jet.b2 * s
    at1[3][0] = obj.jet.b2 * c
    at1[3][1] = obj.jet.b2 * s
    
    # multiply by Sx
    at2 = nulls(4, 4)
    at2[3][3] = - 2 * obj.lep.p
    
    # multiply by (Sx cos + Sy sin)**2
    at3 = nulls(4, 4)
    at3[3][3] = - obj.jet.b2
    return np.array(atc), np.array(at1), np.array(at2), np.array(at3)

class parameters:
    def __init__(self, l, t, z):
        self.l = l
        self.z = z
        self.t = t
        self.htilde = None
        self.hmatrix = None
        self.mobius = None
        self.tag_truth = False
    
    def __str__(self):
        o = "lambda: " + str(self.l) + " Z: " + str(self.z) + " tau: " + str(self.t)
        return o

    def get_plane(self):
        self.A = self.hmatrix.dot([1, 0, 0])
        self.B = self.hmatrix.dot([0, 1, 0])
        self.C = self.hmatrix.dot([0, 0, 1])
        self.normal = np.cross(self.A, self.B)
        return self.normal, np.dot(self.normal, self.C), self.C


class matrix:
    def __init__(self):
        self.RT = None
        self.m_nu = 0
        self.parameters = []

    @property 
    def R_T(self):
        if self.RT is not None: return self.RT
        px, py, pz = self.lep.px, self.lep.py, self.lep.pz
        phi = np.arctan2(py, px)
        theta = np.arctan2(np.sqrt(px**2 + py**2), pz)
        R_z = rotation_z(-phi)
        R_y = rotation_y(0.5*np.pi - theta)
        
        b_vec = np.array([self.jet.px, self.jet.py, self.jet.pz])
        b_rot = R_y @ (R_z @ b_vec)
        R_x = rotation_x(-np.arctan2(b_rot[2], b_rot[1]))
        self.RT = R_z.T @ R_y.T @ R_x.T
        return self.RT

    def mW2(self, Sx):
        a = self.lep.mass ** 2
        b = - 2 * (self.lep.e/self.lep.b)
        c = self.lep.p * (1 - self.lep.b**2) + self.lep.b**2 * Sx
        return a + b * c
    
    def mT2(self, Sx, Sy):
        a = self.mW2(Sx) + self.jet.mass ** 2 
        b = - 2 * self.jet.b * self.jet.e * (Sx * self.cos + Sy * self.sin)
        return a + b

    def get_tau(self, Sx, Sy):
        a = self.o * (Sy - self.w * Sx + self.w * self.lep.e)
        b = self.lep.b * (Sx + self.w * Sy) + self.o ** 2 * self.lep.e
        return atanh(a / b)

    def cache(self):
        self.cos = costheta(self.lep, self.jet)
        self.sin = (1 - self.cos**2) ** 0.5
        self.w   = (self.lep.b/self.jet.b - self.cos)/self.sin
        self.o2  = self.w**2 + 1 - self.lep.b ** 2
        self.o   = self.o2 ** 0.5

        # ------- mapping from psi to theta ------- #
        r = self.lep.b / self.jet.b
        d = ( 1 + self.w**2 - r**2 )**0.5
        self.p_psi_sin = (r * self.w + d) / (1 + self.w**2)
        self.m_psi_sin = (r * self.w - d) / (1 + self.w**2)

        self.p_psi_cos = (r + self.w * d) / (1 + self.w**2)
        self.m_psi_cos = (r - self.w * d) / (1 + self.w**2)

        # ------- Z^2 polynomial -------- #
        self.A = (1 - self.o2)/self.o2
        self.B = -(self.lep.mass/(self.lep.e * self.o))**2
        self.C = 2 * self.w / self.o2
        self.D = 2 * self.lep.p
        self.E = self.lep.mass**2 - self.m_nu**2

        # ------- angular -------- #
        self.cpsi = 1 / (1 + self.w**2)**0.5
        self.spsi = self.w / (1 + self.w**2)**0.5
        self.tpsi = self.w

        # ------- Sx, Sy ------- #
        self.a_x = self.cpsi * self.o / self.lep.b
        self.b_x = - self.spsi 
        self.c_x = - (self.lep.mass ** 2) / self.lep.p

        self.a_y =  self.o * self.spsi / self.lep.b
        self.b_y =  self.cpsi
        self.c_y = -self.w * (self.lep.e**2) / (self.lep.p)

        self.htc, self.ht1, self.ht2 = h_tilde(self)
        self.hc = self.R_T.dot(self.htc)
        self.h1 = self.R_T.dot(self.ht1)
        self.h2 = self.R_T.dot(self.ht2)


        # -------- Lepton Ellipsoid Quadric ------ # 
        self.amc, self.am1, self.am2 = A_mu(self)

        # -------- Jet Ellipsoid Quadric -------- #
        #NOTE: ROTATED IN A_MU FRAME!
        self.bmc, self.bm1, self.bm2, self.bm3 = A_b(self)

    def scan_regions(self, limit = 0.01, get_all = False):
        dg = self.P0_dPdtau0(True)
        for i in sorted(dg):
            t = dg[i]["tau"]
            l = dg[i]["lambda"]
            self.parameters += [parameters(l, t, k) for k in self.PolyZ(l, t)]
       
        dg = self.degenerate_dPdZ()
        for i in dg:
            if abs(dg[i]["discriminant"].imag): continue
            t = dg[i]["tau"].real
            try: 
                l1 = dg[i]["l1"].real
                self.parameters += [parameters(l1, t, k) for k in self.PolyZ(t, l1)]
            except KeyError: pass
            try: 
                l2 = dg[i]["l2"].real
                self.parameters += [parameters(l2, t, k) for k in self.PolyZ(t, l2)]
            except KeyError: pass

        dg = self.degenerate_dPdL()
        for i in dg:
            if abs(dg[i]["discriminant"].imag): continue
            t = dg[i]["tau"].real
            try: 
                l1 = dg[i]["l1"].real
                self.parameters += [parameters(l1, t, k) for k in self.PolyZ(t, l1)]
            except KeyError: pass
            try: 
                l2 = dg[i]["l2"].real
                self.parameters += [parameters(l2, t, k) for k in self.PolyZ(t, l2)]
            except KeyError: pass
       
        dx = {}
        for i in self.parameters:
            sl = self.MobiusTransform(i.t)
            i.tag_truth = self.is_truth
            i.mobius = sl["RHS"] - sl["LHS"]
            i.htilde = self.H_tilde(i.z, i.t)
            i.hmatrix = self.H_matrix(i.z, i.t)
            dx[abs(i.mobius)] = i
       
        if not len(dx): return None
        if min(dx) > limit: self.reject = True
        else: self.reject = False
        return dx[next(iter(sorted(dx)))] if not get_all else self.parameters * (not self.reject)
