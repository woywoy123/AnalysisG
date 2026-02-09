from particle import *
from atomics import *
from original import NuSol
from figures import packet

import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf) 
import math

class debug:
    def __init__(self):
        self.debug_mode = False
        self.plot_mode = True
        self.test_mode = False

    @property
    def splash(self):
        print("--------- pair: " + self.lep.hash + "-" + self.jet.hash + " ---------")
        if self.is_truth: print(">>> TRUTH PAIR <<<")
        else: print("__________ BACKGROUND ____________"); return True
        self.name = self.lep.hash + "-" + self.jet.hash
        return False

    @property 
    def make_truth(self):
        i = Particle(0, 0, 0, 0)
        for j in self.truth_pair:
            if self.lep.hash == j.hash: continue
            if self.jet.hash == j.hash: continue
            i = j
            break

        self.nu  = i
        self.wbs = i + self.lep
        self.top = i + self.lep + self.jet
 
        self.cbW = costheta(self.wbs, self.jet)
        self.cWl = costheta(self.wbs, self.lep)
        self.clb = costheta(self.jet, self.lep)

        self.sbW = (1 - self.cbW**2)**0.5
        self.sWl = (1 - self.cWl**2)**0.5
        self.cWnu = costheta(self.wbs, self.nu)
        self.cphi = (self.clb - self.cbW * self.cWl) / (self.sbW * self.sWl)
        print("=>", angle(self.cphi), angle(self.cbW), angle(self.cWl), angle(self.clb), angle(self.cWnu))

    def debug(self, ev):
        self.make_truth
        ref = NuSol(self.jet, self.lep, self.wbs, self.top, self.nu) 
        assertions("mass W-boson"  , self.mW(ref.Sx, ref.Sy, self.nu.mass), self.wbs.mass)
        assertions("mass Top-Quark", self.mT(ref.Sx, ref.Sy, self.nu.mass), self.top.mass)
 
        assertions("w(+) ", self.wp, ref.w, 0.1, True)
        assertions("O2(+) ", self.op**2, ref.Om2, 0.1, True)
        assertions("Z2(+) ", ref.Z2, self.Z2(ref.Sy, ref.Sx, self.nu.mass, self.wp, self.op), 0.1, True)
    
        ref.use_minus = True
        assertions("w(-) ", self.wm, ref.w, 0.1, True)
        assertions("O2(-) ", self.om**2, ref.Om2, 0.1, True)
        assertions("Z2(-) ", ref.Z2, self.Z2(ref.Sy, ref.Sx, self.nu.mass, self.wm, self.om), 0.1, True)
        ref.use_minus = False
    
        tau = self.Z2_p.tau(ref.Sx, ref.Sy)
        Sy, Sx = self.Z2_p.Sy(tau, ref.Z), self.Z2_p.Sx(tau, ref.Z)
        assertions("Sx(+) ", ref.Sx - self.Z2_p.Sx0, Sx - self.Z2_p.Sx0, 0.1, True)
        assertions("Sy(+) ", ref.Sy - self.Z2_p.Sy0, Sy - self.Z2_p.Sy0, 0.1, True)
        assertions("Z2(+) ", ref.Z2, self.Z2(ref.Sy, ref.Sx, self.nu.mass, self.wp, self.op), 0.1, True)
        assertions("SxSy (+)", np.array([[ref.Sx], [ref.Sy]]), self.Z2_p.SxSy(tau, ref.Z), 0.1, True)

        tau = self.Z2_pm.tau(ref.Sx, ref.Sy)
        Sy, Sx = self.Z2_pm.Sy(tau, ref.Z), self.Z2_pm.Sx(tau, ref.Z)
        assertions("Sx(+-) ", ref.Sx - self.Z2_pm.Sx0, Sx - self.Z2_pm.Sx0, 0.1, True)
        assertions("Sy(+-) ", ref.Sy - self.Z2_pm.Sy0, Sy - self.Z2_pm.Sy0, 0.1, True)
        assertions("Z2(+-) ", ref.Z2, self.Z2(ref.Sy, ref.Sx, self.nu.mass, self.wp, self.op), 0.1, True)
        assertions("SxSy (+-)", np.array([[ref.Sx], [ref.Sy]]), self.Z2_pm.SxSy(tau, ref.Z), 0.1, True)

        Sy, Sx = self.dG2_SySx(0.1)
        z2p = self.Z2(Sy, Sx, 0, self.wp, self.op)
        z2m = self.Z2(Sy, Sx, 0, self.wm, self.om)
        assertions("dG2", 1 + z2p - z2m - self.dG2(Sy, Sx), 1.0, True)
        assertions("Z2mm", self.Z2f(Sx, 0, -1), self.Z2(Sx / self.dm, Sx, 0, self.wm, self.om), 0.1, True)
        assertions("Z2mp", self.Z2f(Sx, 0, -1), self.Z2(Sx / self.dm, Sx, 0, self.wp, self.op), 0.1, True)
        assertions("Z2pp", self.Z2f(Sx, 0, +1), self.Z2(Sx / self.dp, Sx, 0, self.wp, self.op), 0.1, True)
        assertions("Z2pm", self.Z2f(Sx, 0, +1), self.Z2(Sx / self.dp, Sx, 0, self.wm, self.om), 0.1, True)

        try: 
            Syi, Sxi = self.Z2_p.Sy(-0.1), self.Z2_p.Sx(-0.1)
            tau = self.Z2_p.tau(Sxi, Syi)
            Syo, Sxo = self.Z2_p.Sy(tau), self.Z2_p.Sx(tau)
            assertions("Sx (+)", Sxi, Sxo, 0.1, True)
            assertions("Sy (+)", Syi, Syo, 0.1, True)
        except np.linalg.LinAlgError: pass

        try: 
            Syi, Sxi = self.Z2_m.Sy(-0.1), self.Z2_m.Sx(-0.1)
            tau = self.Z2_m.tau(Sxi, Syi)
            Syo, Sxo = self.Z2_m.Sy(tau), self.Z2_m.Sx(tau)
            assertions("Sx (-)", Sxi, Sxo, 0.1, True)
            assertions("Sy (-)", Syi, Syo, 0.1, True)
        except np.linalg.LinAlgError: pass

        # accounting for floating point precision
        assertions("Sx+ Tangent", 0, self.dZ2dSx(self.dZ2dSxSy_max(+1)[0], +1), 1, True) 
        assertions("Sx- Tangent", 0, self.dZ2dSx(self.dZ2dSxSy_max(-1)[0], -1), 1, True)


        # ----------- testing the factored Sx^2 coefficients ------------------ %
        sn = +1
        pre_c = self.Z2f(0, 0, sn, True)
        tru_c = self.Z2(0, 0, 0, signs(self.wp, self.wm, sn),  signs(self.op, self.om, sn), True)
        ps = signs(self.t_psi_p, self.t_psi_m, sn)
        assertions("(+)Sx^2 coefficient", (tru_c[0] + tru_c[1] / ps + tru_c[2] / ps ** 2), pre_c[0], 0.001, True) 
        assertions("(+)Sx coefficient"  , tru_c[3], pre_c[1], 0.000001, True)
        assertions("(+)C coefficient"   , tru_c[4], pre_c[2], 0.000001, True)

        sn = -1
        pre_c = self.Z2f(0, 0, sn, True)
        tru_c = self.Z2(0, 0, 0, signs(self.wp, self.wm, sn),  signs(self.op, self.om, sn), True)
        ps = signs(self.t_psi_p, self.t_psi_m, sn)
        assertions("(-)Sx^2 coefficient", (tru_c[0] + tru_c[1] / ps + tru_c[2] / ps ** 2), pre_c[0], 0.001, True)
        assertions("(-)Sx coefficient"  , tru_c[3], pre_c[1], 0.000001, True)
        assertions("(-)C coefficient"   , tru_c[4], pre_c[2], 0.000001, True)
       
    def test_bench(self):
        def plus(u): return "+" if u > 0 else "-"

        self.make_truth
        ref = NuSol(self.jet, self.lep, self.wbs, self.top, self.nu) 
        sols = ref.solution #["neutrino"]

        print(self.dG2_SySx(0.1))
        print(self.dm)
        exit()
        psi_p, psi_m = self.psi_p, self.psi_m
        print(psi_p, psi_m)
        phi = (1 / 2) * (psi_p + psi_m)
        psi = (1 / 2) * (psi_p - psi_m)


        print(phi)
        print(psi)
        exit()

        print(self.lep)
        print(self.jet)
        S = self.dG2_SySx(0.1)

        tau = 0.1
        f = (( np.cos(psi_p) * np.cos(psi_m)) / ( abs(self.Gamma(+1) * self.Gamma(-1) )))**0.5
        Sx = -f * (math.tan(phi)                  - math.tan(psi) * math.tanh(tau)) * math.cosh(tau) * math.cos(phi)
        Sy =  f * (math.tan(phi) * math.tanh(tau) +                  math.tan(psi)) * math.cosh(tau) * math.sin(phi)
        print(S[1], Sx, S[0], Sy)
        print(self.dG2(Sy, Sx), self.dG2(S[0], S[1]))

        exit()

        data = []
        for i in range(1000000):
            tau = 0.0001 * ( i - 10000)
            S = self.dG2_SySx(tau)
            #data.append([tau, S[0] / S[1], self.Z2cLine(S[-1], +1) - self.Z2cLine(S[-1], -1)])
            #print(data[-1])
            #if tau > 10: break
            #f = [1, -1]
            #K = [[f[i], f[j], f[k], f[t]] for i in range(2) for j in range(2) for k in range(2) for t in range(2)]
            #vis = packet(sols["H_T"], sols["neutrino"], inst = self)
            #for ix in K:
            #    c, o = self.clines(S[1], S[1], ix[0], ix[1], ix[2], ix[3]) 
            #    if not c: continue
            #    key = plus(ix[0]) + plus(ix[1]) + plus(ix[2]) + plus(ix[3])
            #    for k in o: 
            #        print(k)
            #        vis.add_line(k, key + "(" + k.name + ")", k.sol_pts, (-1000000, 1000000))
            #vis.compile3D()



    def figures(self):
        self.make_truth
        ref = NuSol(self.jet, self.lep, self.wbs, self.top, self.nu) 
        data = ref.solution

        #vis = packet(None, np.array([ref.Sx, ref.Sy])) 
        #vis.add_hyperbolic(self.G2_p.SxSy, "G2-plus" , domain = (-16, 16))
        #vis.add_hyperbolic(self.G2_m.SxSy, "G2-minus", domain = (-16, 16))
        #vis.add_hyperbolic(self.Z2_pp.SxSy, "Z2- (++)", domain = (-8, 8))
        #vis.add_hyperbolic(self.Z2_mp.SxSy, "Z2- (-+)", domain = (-8, 8))
        #vis.add_hyperbolic(self.Z2_pm.SxSy, "Z2- (+-)", domain = (-6, 6))
        #vis.add_hyperbolic(self.Z2_mm.SxSy, "Z2- (--)", domain = (-3, 3))
        #vis.compile2D() #"test")
        
        def chi2(tru, cand): return (sum((tru - cand.sol_pts)**2)**0.5)*0.001
        def best(tru, lst, dct): 
            for i in lst: dct[chi2(tru, i)] = i
            return dct
        
        def search(tru, Sx1, Sx2, out):
            f = [1, -1]
            K = [[f[i], f[j], f[k], f[t]] for i in range(2) for j in range(2) for k in range(2) for t in range(2)]
            for i in K: out = best(tru, self.clines(Sx1, Sx2, i[0], i[1], i[2], i[3])[-1], out)
            return out

        tru = data["neutrino"]
        pkt = packet(data["H_T"], data["neutrino"], inst = self)
        Sx11,  Sy11 = self.dZ2dSxSy_max(+1)
        Sx22,  Sy22 = self.dZ2dSxSy_max(-1)
        out = {}

        cand = self.Z2_crootL(+1, True) + self.Z2_crootL(-1, True) + [Sx11, Sx22]
        for i in cand: 
            for j in cand: out = search(tru, i, j, out)

        ls = sorted(out)
        acpt = []
        for i in range(len(ls)):
            x = ls[i]
            tmp = {}
            for j in range(len(ls)):
                if i >= j: continue
                y = abs(ls[j] / ls[i])
                tmp[y] = [ls[j], ls[i]]
            if not len(tmp): continue
            cl = sorted(tmp)[0]
            a, b = tmp[cl] 
            if a < b: continue
            ok = True
            for k in acpt:
                if abs((b - k) / (b + k)) > 1e-3: continue
                ok = False; break
            if ok: acpt.append(ls[i])
        lp = [out[i] for i in acpt]
        print("++++++++++++++> ", ls[:6])

        def as_vec(l1, l2):
            ux = [l1.x0, l1.y0, l1.sol_pts[-1]]
            vx = [l2.x0, l2.y0, l2.sol_pts[-1]]
            zx =   ux[1] * vx[2] - vx[1] * ux[2]
            zy = -(ux[0] * vx[2] - vx[0] * ux[2])
            zz =   ux[0] * vx[1] - vx[0] * ux[1]
            return np.array([zx, zy, zz]), np.array(ux), np.array(vx)

        # find the tilt - probably something extremely trivial which I will regret checking
        for i in range(len(lp)):
            for j in range(len(lp)):
                if i >= j: continue
                zv, ux, vx = as_vec(lp[i], lp[j])
                AxB = sum([k**2 for k in zv])
                A   = sum([k**2 for k in ux])
                B   = sum([k**2 for k in vx])
                s   = AxB / (A * B)
                lp[i].rz = - math.asin(s)
                ztx = lp[i].Rz.T.dot(ux.reshape((3, 1)))
                ztx = ztx.T.reshape(3)
                lp[i].ry = - math.atan2(ztx[-1], ztx[0])
                ztx = lp[i].Ry.T.dot(ztx.reshape((3, 1))).T.reshape(3)
                lp[i].rx =  -math.atan2(ztx[-1], ztx[1])


        for i in range(0, len(lp) if len(lp) < 12 else 12):
            HxP = self.HR_tilde(lp[i].x0,  lp[i].y0, lp[i].sol_pts[-1], lp[i].sign1)
            hx = HxP[:,2].reshape(3, 1)
            ar = np.array(([[1., 0.], [0., 1.], [0., 0.]]))

            dp = np.diag([self.delta(+1), self.delta(-1), - self.delta(-1) * self.delta(+1)])

            #HxP     = lp[i].Rz.T.dot(HxP - np.concatenate((ar,hx), -1)).dot(lp[i].Rz.T) 
            sol_pts = HxP[:,2].reshape(3, 1)
            HxP  = (dp.T).T.dot(np.linalg.inv(HxP - dp)).dot(dp)
            pkt.add_ellipse(HxP, "H[PMPM]" + str(i), sol_pts)
        pkt.compile2D_Proj(None, True) #None, False)
    
        return 
        data = ref.solution
        pkt = packet(data["H_T"], data["neutrino"], inst = self)

        Sx11,  Sy11 = self.dZ2dSxSy_max(+1)
        Sx22,  Sy22 = self.dZ2dSxSy_max(-1)

        l = self.lines(Sx11, Sy11, Sx22, Sy22)
        SxP, SyP, Z2p = self.SolveSxSy(l["plus"]["x0"] ,  l["plus"]["y0"], m_nu_p, +1)
        SxM, SyM, Z2m = self.SolveSxSy(l["minus"]["x0"], l["minus"]["y0"], m_nu_m, -1) 

        HxP = self.HR_tilde(l["plus"]["x0"] ,  l["plus"]["y0"], abs(Z2p)**0.5, +1)
        HxM = self.HR_tilde(l["minus"]["x0"], l["minus"]["y0"], abs(Z2m)**0.5, +1)

        pkt.add_ellipse(HxP, "H1111pp", np.array((l["plus"]["x0"] ,  l["plus"]["y0"], abs(Z2p)**0.5)), False)
        pkt.add_ellipse(HxM, "H1111mp", np.array((l["minus"]["x0"], l["minus"]["y0"], abs(Z2m)**0.5)), False)

        _SxP, _SyP, _Z2p = self.SolveSxSy(l["plus"]["x0"] ,  l["plus"]["y0"], m_nu_p, -1)
        _SxM, _SyM, _Z2m = self.SolveSxSy(l["minus"]["x0"], l["minus"]["y0"], m_nu_m, +1) 

        HxP = self.HR_tilde(l["plus"]["x0"] ,  l["plus"]["y0"], abs(_Z2p)**0.5, -1)
        HxM = self.HR_tilde(l["minus"]["x0"], l["minus"]["y0"], abs(_Z2m)**0.5, -1)

        pkt.add_ellipse(HxP, "H1111pm", np.array((l["plus"]["x0"] ,  l["plus"]["y0"], abs(_Z2p)**0.5)), False)
        pkt.add_ellipse(HxM, "H1111mm", np.array((l["minus"]["x0"], l["minus"]["y0"], abs(_Z2m)**0.5)), False)

        pkt.compile3D()






