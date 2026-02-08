from variables import *
from atomics import *

class Conuix(variables):

    def __init__(self, lep, jet, event):
        self.lep, self.jet = lep, jet
        self.truth_pair = []
        self.is_truth  = True
        self.is_truth *= (jet.top_index == lep.top_index)
        self.is_truth *= (lep.top_index in event.truth_pairs)
        self.truth_pair = event.truth_pairs[jet.top_index] if self.is_truth else []
        if self.splash: return
        variables.__init__(self, lep, jet)

        if self.debug_mode: self.debug(event)
        if self.plot_mode: self.figures()
        if self.test_mode: self.test_bench()

    def mW(self, Sx, Sy, m_nu):
        return complex(m_nu**2 - self.m_mu**2 - 2 * self.p_mu * Sx) ** 0.5

    def mT(self, Sx, Sy, m_nu):
        a = self.m_b**2 - self.m_mu ** 2 + m_nu ** 2 
        b = - 2 * (Sx * (self.p_mu + self.p_b * self.c_th) + self.p_b * self.s_th * Sy)
        return complex(a + b)**0.5

    def Z2f(self, Sx, m_nu, sign, get_coeff = False):
        a = - (1 + self.t_th / signs(self.t_psi_p, self.t_psi_m, sign))
        b = 2 * self.p_mu 
        c = self.m_mu ** 2 - m_nu ** 2 
        if get_coeff: return [a, b, c]
        return a * Sx**2 + b * Sx + c 

    def Z2(self, Sy, Sx, m_nu, w, o, get_coeff = False):
        a = (self.b_mu ** 2 - w ** 2) / o ** 2 
        b = 2 * w / o ** 2
        c = - (1 - self.b_mu ** 2) / o ** 2 
        d = 2 * self.p_mu  
        e = self.m_mu ** 2 - m_nu ** 2 
        co = iter([a, b, c, d, e])
        fo = iter([ Sx**2, Sx * Sy,  Sy ** 2, Sx, 1])
        if get_coeff: return [a, b, c, d, e]
        return sum([next(co) * next(fo) for _ in range(5)])

    # -------- Constants generators ----------- #
    def omega(self, sign):
        return (1 / self.s_th) * ( sign * self.b_mu / self.b_b - self.c_th)

    def Omega(self, sign): 
        return (signs(self.wp, self.wm, sign) ** 2 + 1 - self.b_mu ** 2)**0.5

    # -------- Eigenvalues ------ #
    def dG2_lambda(self, sign):
        f = - self.Gp * self.Gm / (2 * self.c_psi_p * self.c_psi_m)
        return f * (math.cos(self.psi_p - self.psi_m) - sign)

    def Gamma(self, sign):
        return (self.wp + sign * self.wm) / signs(self.op, self.om, sign)**2


    def komega(self):
        r = self.b_mu / self.b_b
        kp, km = r + 1, r - 1
        tpx = math.tan(self.theta)
        kmap = 0.5 * np.array([[kp,   km], [-km, -kp]])
        kpsi = 0.5 * np.array([[tpx,  1 / tpx], [-1 / tpx, -tpx]])
        print(np.linalg.inv(kmap).dot(kpsi))


        
    # --------- delta G^2 factorization parameters --------- # 
    def dG2_factorization(self):
        self.psi_p  , self.psi_m   = math.atan(self.dp)  , math.atan(self.dm)
        self.c_psi_p, self.c_psi_m = math.cos(self.psi_p), math.cos(self.psi_m) 
        self.s_psi_p, self.s_psi_m = math.sin(self.psi_p), math.sin(self.psi_m) 
        self.t_psi_p, self.t_psi_m = math.tan(self.psi_p), math.tan(self.psi_m) 

        #self.eigs = 2 * (0.5 - (self.Gp * self.Gm > 1))
        self.phi  = - (1 / 2) * (self.psi_p + self.psi_m)
        self.psi  =   (1 / 2) * (self.psi_p - self.psi_m)


        # ---------- delta G^2 eigenvalues ------------------- #
        self.lam_p, self.lam_m = abs(self.dG2_lambda(+1)), abs(self.dG2_lambda(-1))

        self.vp = np.array([[  math.cos(self.phi), math.sin(self.phi)]]).T
        self.vm = np.array([[- math.sin(self.phi), math.cos(self.phi)]]).T
        self.eigv = np.concatenate((self.vp, self.vm), -1)
        self.eigm = self.eigv.dot(np.array([
            [1 / math.sqrt(self.lam_m),                         0], 
            [0                        , 1 / math.sqrt(self.lam_p)]
        ]))

        
    def dG2_SySx(self, tau):
        S = self.eigm.dot(np.array([[math.cosh(tau), math.sinh(tau)]]).T) 
        return S[1][0], S[0][0]

    # --------- Conditional where the two branches become equal -------- #
    def dG2(self, Sy, Sx):
        return - self.Gp * self.Gm * (Sx - self.dp * Sy) * (Sx - self.dm * Sy)

    # -------- A refactored version where the cline meet ----------------- #
    def Z2cLine(self, Sx, sign):
        m_nu = self.mass_neutrino(sign)     
        return -( (1 + self.t_th / signs(self.t_psi_p, self.t_psi_m, sign)) * Sx ** 2 + 2 * Sx * self.p_mu + self.m_mu ** 2 - m_nu ** 2)
       
    # -------- Derivative of Sx at lines --------- #
    def dZ2dSx(self, Sx, sign):
        return - 2 * ( (1 + self.t_th / signs(self.t_psi_p, self.t_psi_m, sign)) * Sx - self.p_mu)

    def mass_neutrino(self, sign):
        tn = signs(self.t_psi_p, self.t_psi_m, sign) 
        return complex(self.m_mu ** 2 + (self.p_mu ** 2 * tn) / (tn + self.t_th))**0.5 

    # -------- Roots of dG2 = Z^2+ - Z^2- -> 0 --------- #
    def delta(self, sign):
        a = (self.op + float(sign) * self.om) ** 2  - (self.wp + self.wm) ** 2
        b = 2 * (self.wp + self.wm)
        return a / b
 
    # ---------- reverse mapping (x1, y1) -> Sx, Sy ------------- #
    def SolveSxSy(self, x1, y1, m_nu, sign):
        w = signs(self.wp, self.wm, sign)
        Sx = -(1 / self.b_mu ** 2) * ( w * y1 +      (1 - self.b_mu ** 2)*x1 )
        Sy = -(1 / self.b_mu ** 2) * ( w * x1 + (w ** 2 - self.b_mu ** 2)*y1 )
        return Sx, Sy, self.Z2(Sx, Sy, m_nu, w, signs(self.op, self.om, sign)) 


    # --------------- paper matrices ---------------- #
    def x1(self, Sx, Sy, sign):
        w = signs(self.wp, self.wm, sign)
        return Sx - (Sx + w * Sy) / signs(self.op, self.om, sign)**2

    def y1(self, Sx, Sy, sign):
        w = signs(self.wp, self.wm, sign)
        return Sy - (Sx + w * Sy) * w / signs(self.op, self.om, sign)**2

    @property 
    def R_T(self):
        if self.RT is not None: return self.RT
        px, py, pz = self.lep.px, self.lep.py, self.lep.pz
        phi   = np.arctan2(py, px)
        theta = np.arctan2(np.sqrt(px**2 + py**2), pz)
        R_z   = rotation_z(-phi)
        R_y   = rotation_y(0.5*np.pi - theta)
        
        b_vec = np.array([self.jet.px, self.jet.py, self.jet.pz])
        b_rot = R_y @ (R_z @ b_vec)
        R_x = rotation_x(-np.arctan2(b_rot[2], b_rot[1]))
        self.RT = R_z.T @ R_y.T @ R_x.T
        return self.RT

    def H_tilde(self, Sy, Sx, m_nu, sign): 
        w = signs(self.wp, self.wm, sign)
        o = signs(self.op, self.om, sign)
        Z = complex(self.Z2(Sy, Sx, m_nu, w, o)) ** 0.5
        Z = Z.real
        g = np.array([
            [Z / o    , 0, self.x1(Sx, Sy, sign) - self.p_mu ],
            [Z * w / o, 0, self.y1(Sx, Sy, sign)             ],
            [0        , Z,                                  0],
        ])
        return g.real

    def HR_tilde(self, x1, y1, z1, sign):
        w = signs(self.wp, self.wm, sign)
        o = signs(self.op, self.om, sign)
        g = np.array([
            [z1 / o    , 0, x1],
            [z1 * w / o, 0, y1],
            [0        , z1,  0],
        ])
        return g.real






class Conuic:
    def __init__(self, particles, event):
        self.lep, self.jet = [], []
        for i in particles:
            l = self.lep if i.mass < 200 else self.jet
            l.append(i)
        self.engine = [Conuix(i, j, event) for i in self.lep for j in self.jet]
 
