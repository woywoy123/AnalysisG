from atomics import *
import cmath

def AQuad(data, s1, s2, m_nu = None):
    d = delta(data, s2)
    w = omega(data, s1)
    o = Omega(data, s1)
    b, p, m = data.b_mu, data.p_mu, data.m_mu
    if m_nu is None: m_nu = nu_mass(data, s1, s2)

    a_ = ((b**2 - w **2 )* d ** 2 + 2 * w * d - (1 - b **2 ))/o**2
    b_ = 2 * p * d 
    c_ = m**2 - 2 * m_nu**2
    r1 = (-b_ + complex(b_**2 - 4 * a_ * c_)**0.5)/(2 * a_)
    r2 = (-b_ - complex(b_**2 - 4 * a_ * c_)**0.5)/(2 * a_)
    return {"SP" : [m_nu, r1 / d, r1], "SM" : [m_nu, r2 / d, r2]}

class data_t:
    def __init__(self, jet, lep):
        self.theta = angular_t(math.acos(costheta(jet, lep)))

        self.m_mu = lep.mass
        self.b_mu = lep.b
        self.p_mu = lep.p
        self.e_mu = lep.e
        
        self.m_b = jet.mass
        self.b_b = jet.b
        self.p_b = jet.p
 
        kp = (self.b_mu / self.b_b) + 1 
        km = (self.b_mu / self.b_b) - 1 
        psi = self.theta.alpha / 2
        self.k_vector = np.array([kp, km]).reshape((2, 1))
        
        self.w_matrix = (1 / 2) * np.array([
            [     math.tan(psi), 1 / math.tan(psi)],
            [-1 / math.tan(psi),    -math.tan(psi)]
        ])

class Z2_t:
    def __init__(self, data, sign, eps):
        self.w = omega(data, sign)
        self.o = Omega(data, sign)

        self.b_mu = data.b_mu
        self.psi = angular_t(math.atan(self.w))

        self.a = (data.b_mu ** 2 - self.w ** 2) / self.o**2
        self.b = 2 * self.w  / self.o**2
        self.c = - (1 - data.b_mu**2) / self.o**2
        self.d = 2 * data.p_mu
        self.e = data.m_mu ** 2 

        self.Sx0 = - (data.m_mu ** 2) / (data.p_mu)
        self.Sy0 = - self.w * data.e_mu ** 2 / data.p_mu
        self.Sz0 = self.Z2(self.Sx0, self.Sy0, 0) 

        # ----------------- eigenvalues ------------- #
        self.kappa = angular_t(math.atan(self.w))
        self.ep = (data.b_mu / self.o)**2
        self.em = -1

        self.data = data
        self.xkp = sign
        self.eps = eps # <---- hyperbolic branches
    
    def Z2(self, sx, sy, m_nu = None):
        if m_nu is None: m_nu = self.Sz0 ** 0.5
        ls = [self.a, self.b, self.c, self.d, self.e - m_nu**2]
        Sp = [sx**2, sx*sy, sy**2, sx, 1]
        return sum([i * j for i, j in zip(ls, Sp)])

    def Sx(self, tau, m_nu = None):
        tx = hyper_t(tau)
        ep = math.sqrt(1 / self.ep)
        if m_nu is None: m_nu = self.Sz0 ** 0.5
        return m_nu * self.kappa.cos * (self.eps * ep * tx.cosh - self.kappa.tan * tx.sinh) + self.Sx0

    def Sy(self, tau, m_nu = None):
        tx = hyper_t(tau)
        ep = math.sqrt(1 / self.ep)
        if m_nu is None: m_nu = self.Sz0 ** 0.5
        return m_nu * self.kappa.cos * (self.eps * ep * tx.cosh * self.kappa.tan + tx.sinh) + self.Sy0

    def NuVec(self, tau, phi, m_nu = None):
        hx, ax = hyper_t(tau), angular_t(phi)
        Sx = self.eps * self.o * self.psi.cos * hx.cosh - self.b_mu * ax.cos * self.psi.sin * hx.sinh
        Sy = self.eps * self.o * self.psi.sin * hx.cosh + self.b_mu * ax.cos * self.psi.cos * hx.sinh
        Sx, Sy = (m_nu / self.b_mu) * Sx + self.Sx0, (m_nu / self.b_mu) * Sy + self.Sy0
        Sz = self.eps * m_nu * hx.sinh * ax.sin
        return [Sx, Sy, Sz] 

class G2_t:
    def __init__(self, data):
        self.w = branch_t(omega(data,+1), omega(data,-1))
        self.o = branch_t(Omega(data,+1), Omega(data,-1))
        self.G = branch_t(Gamma(data,+1), Gamma(data,-1))

        self.delta = branch_t(delta(data,+1), delta(data,-1))
        self.psi   = branch_t(angular_t(math.atan(self.delta.p)), angular_t(math.atan(self.delta.m)))
        self.phi   = angular_t((self.psi.p.alpha + self.psi.m.alpha)/2)
        self.alpha = angular_t((self.psi.p.alpha - self.psi.m.alpha)/2)

        f = (self.G.m * self.G.p / (2 * self.psi.p.cos * self.psi.m.cos))
        self.ep = f * (-math.cos(self.psi.p.alpha - self.psi.m.alpha) + 1)
        self.em = f * (-math.cos(self.psi.p.alpha - self.psi.m.alpha) - 1)

        self.det = - (self.G.m * self.G.p / 2)**2 * (math.sin(self.psi.p.alpha - self.psi.m.alpha)**2)
        self.det =   self.det / (self.psi.p.cos**2 * self.psi.m.cos**2)

    def Mobius(self, sx, sy):
        return (sx - self.delta.m * sy) / (sx - self.delta.p*sy)

    def G2(self, sx, sy):
        return -self.G.m * self.G.p * (sx - self.delta.m * sy)*(sx - self.delta.p * sy)

    def Sx(self, K, tau):
        tx = hyper_t(tau)
        r = ((K * self.psi.m.cos * self.psi.p.cos)/(self.G.m * self.G.p))**0.5
        return r * ((self.phi.cos / self.alpha.sin) * tx.cosh - (self.phi.sin / self.alpha.cos) * tx.sinh)

    def Sy(self, K, tau):
        tx = hyper_t(tau)
        r = ((K * self.psi.m.cos * self.psi.p.cos)/(self.G.m * self.G.p))**0.5
        return r * ((self.phi.sin / self.alpha.sin) * tx.cosh + (self.phi.cos / self.alpha.cos) * tx.sinh)

    def P0(self, tau, sign):
        w = signs(self.wp, self.wm, sign)
        o = signs(self.op, self.om, sign)
        phi = math.atan(w)
        alphap = o * math.sin(phi) + self.lep.b * math.cos(phi) * math.tanh(tau)
        alpham = o * math.cos(phi) - self.lep.b * math.sin(phi) * math.tanh(tau)
        return alpham  * self.lep.b * math.cos(phi) * math.sqrt(1 - math.tanh(tau)) + alphap / alpham




