from conuix.base.atomics import *
from conuix.types.structs import *

class attestation:
    def __init__(self): self.verbose = True

    def _check(self, name, v1, v2, sign = 0, lm = 0.001, hard = True): 
        if sign == 0: return assertions(name, v1, v2, lm, hard, self.verbose)
        name += " [" + ["+", "-"][sign < 0] + "] "
        self._check(name, v1, v2, 0, lm, hard)

    # --- test the base relations --- #
    def proof_base_relation(self):
        pl , mn  = self.pls, self.mns
        psx, psy = pl.Sx, pl.Sy
        msx, msy = mn.Sx, mn.Sy
        m_nu = self.nu.mass
        self.verbose = False

        # -------- Basic mass relations e.t.c ------ #
        # Mainly designed to verify that all expressions 
        # have been properly implemented.
        mw2, mt2 = self.mW2(psx, m_nu), self.mT2(psx, psy, m_nu)  
        self._check("mW2", pl.mW2, mw2, 0.00001, True)
        self._check("mT2", pl.mT2, mt2, 0.00001, True)

        self._check("w", pl.w, self.wp, +1)
        self._check("w", mn.w, self.wm, -1)

        self._check("O", pl.Om2, self.Op**2, +1)
        self._check("O", mn.Om2, self.Om**2, -1)

        z2pv, z2pt = self.Z2(psx, psy, m_nu, +1)[0], pl.Z2
        z2mv, z2mt = self.Z2(msx, msy, m_nu, -1)[0], mn.Z2

        self._check("Z2", z2pt, z2pv, +1)
        self._check("Z2", z2mt, z2mv, -1)

        # ------ Test transformation ------ #
        PP = self.Q2(psx, psy, 0,  +1, m_nu)
        self._check("Q2 -> Z2", z2pt, PP, +1)
        MM = self.Q2(psx, psy, 0, -1, m_nu)
        self._check("Q2 -> Z2", z2mt, MM, -1)

        sx0, sy0 = self.to_Sx0(0), self.to_Sy0(0, +1)
        self._check("Z2(S0) = - m_nu^2", self.Z2(sx0, sy0, m_nu, +1)[0],   - m_nu ** 2, +1)
        self._check("Q2(S0) = - m_nu^2", self.Q2(sx0, sy0,    0, +1, m_nu), -m_nu ** 2, +1)

        sx0, sy0 = self.to_Sx0(0), self.to_Sy0(0, -1)
        self._check("Z2(S0) = - m_nu^2", self.Z2(sx0, sy0, m_nu, -1)[0],   - m_nu ** 2, -1)
        self._check("Q2(S0) = - m_nu^2", self.Q2(sx0, sy0,    0, -1, m_nu), -m_nu ** 2, -1)

        # ----- Recover neutrino kinematics from truth ---- #
        nvec = self.F_frame(self.nu)
        zp = z2pt**0.5

        cchi = np.acos( (self.Op / zp) * (nvec[0] - self.x1(psx, psy, +1) + self.p_mu) )
        schi = np.asin(nvec[2] / zp)
        s, c = schi > 0, cchi > 0
        if s     and     c: chi = cchi
        if s     and not c: chi = schi
        if not s and not c: chi = np.pi - schi
        if not s and     c: chi = -cchi

        vpt = self.vec_nu(psx, psy, zp, chi, +1)
        self._check("Fnu_px", nvec[0], vpt[0]) 
        self._check("Fnu_py", nvec[1], vpt[1]) 
        self._check("Fnu_pz", nvec[2], vpt[2]) 
        self._check("Fnu_pe", nvec[3], vpt[3]) 

    def proof_pcl1_relation(self):
        psx, psy = self.pls.Sx, self.pls.Sy 
        m_nu = self.nu.mass
        
        z2p = self.Z2(psx, psy, m_nu, +1)[0]
        z2m = self.Z2(psx, psy, m_nu, -1)[0]
        
        self._check("dG2", z2p - z2m, self.G2(psx, psy))
        self._check("d[+]d[-] = -(1 - beta^2_mu)", -(1 - self.b_mu**2), self.dp * self.dm)
        self._check("d[-] =  tanh(phi) * sqrt(1 - beta^2)", self.dm, np.tanh(self.delta_phi) * np.sqrt(1 - self.b_mu**2))
        self._check("d[+] = -coth(phi) * sqrt(1 - beta^2)", self.dp, -np.tanh(self.delta_phi)**-1 * np.sqrt(1 - self.b_mu**2))

        lp, lm = self.to_Lp(psx, psy), self.to_Lm(psx, psy)        
        sx, sy = self.to_Sx(lp, lm), self.to_Sy(lp, lm)        
        
        self._check("S -> L -> S", psx, sx, +1)
        self._check("S -> L -> S", psy, sy, -1)
    
        z2lp = self.Z2L(lp, lm, m_nu, +1)
        z2lm = self.Z2L(lp, lm, m_nu, -1)
        
        self._check("Z2 -> Z2L", z2p, z2lp, +1)
        self._check("Z2 -> Z2L", z2m, z2lm, -1)

        self._check("(crit) Z2L -> L", self.Z2L(self.to_Lp0(0, +1), self.to_Lm0(0, +1), m_nu, +1), - m_nu**2, +1)
        self._check("(crit) Z2L -> L", self.Z2L(self.to_Lp0(0, -1), self.to_Lm0(0, -1), m_nu, -1), - m_nu**2, -1)

        lb = lambda_t(self) 
        lp, lm = lb.Lx(psx, psy), lb.Ly(psx, psy)        
        sx, sy = lb.Sx(lp, lm), lb.Sy(lp, lm)        
        self._check("S -> L -> S", psx, sx, +1)
        self._check("S -> L -> S", psy, sy, -1)
        self._check("Z2N", z2p, lb.Z2P(lp, lm, m_nu), +1)
        self._check("Z2N", z2m, lb.Z2M(lp, lm, m_nu), -1)
        self._check("det(M[+] - M[-]) = 0", 1.000, 1 - np.linalg.det((self.MQ(+1) - self.MQ(-1))))
        
    def proof_pcl1_eigen(self):
        sx, sy = self.pls.Sx, self.pls.Sy 
        m_nu = self.nu.mass
         
        z2p = self.Z2(sx, sy, m_nu, +1)[0]
        z2m = self.Z2(sx, sy, m_nu, -1)[0]
        dZ2 = z2p - z2m

        ap , am  = self.to_Ap(sx, sy), self.to_Am(sx, sy)
        sx_, sy_ = self.A_Sx(ap[0], am[0]),  self.A_Sy(ap[0], am[0])
        self._check("Alpha(Sx, Sy) -> (Sx, Sy)", sx, sx_, +1)
        self._check("Alpha(Sx, Sy) -> (Sx, Sy)", sy, sy_, -1)
        self._check("dZ2", dZ2, ap[1] * ap[0] ** 2 + am[1] * am[0] ** 2)


    def proof_pcl2_relation(self):
        sx, sy = self.pls.Sx, self.pls.Sy 
        m_nu = self.nu.mass
        MP, MM = self.MQ(+1) - np.diag([0, 0, 0, m_nu**2]), self.MQ(-1) - np.diag([0, 0, 0, m_nu**2]) 
        l1, l2, l3, l4 = self.Pl2(m_nu) 
        nu = self.F_frame(self.nu)

        self._check("det(Q[+] Q[-]^-1) = (O[-] / O[+])^2 ", (self.Om / self.Op) ** 2, np.linalg.det(MP.dot(np.linalg.inv(MM))) ) 
        self._check("det(Q[+]^-1 Q[-]) = (O[+] / O[-])^2 ", (self.Op / self.Om) ** 2, np.linalg.det(MM.dot(np.linalg.inv(MP))) ) 
        self._check("l1 x l2 = (O[-] / O[+])^2",  (self.Om / self.Op) ** 2, l1 * l2)
        
        eig = np.linalg.eigvals(np.linalg.solve(MP, MM))
        eig = [i for i in eig if i.real != 1.0]
        lx = []
        for i in [l1, l2, l3, l4]: 
            mx = -1
            for j in eig: 
                x = abs(i - j)**2
                if mx < 0 or mx > x: mx = x
            if mx > 0.0001: continue
            lx.append(i)
        assert len(lx) >= 2
        
        m_nu = self.nu.mass ** 2
        l1, l2, l3, l4 = self.Pl2(complex(m_nu) ** 0.5) 
        self._check("m^2_nu(lambda)", self.Pl2_m2(l1, +1), m_nu, +1) 
        self._check("m^2_nu(lambda)", self.Pl2_m2(l2, +1), m_nu, +1) 
        self._check("m^2_nu(lambda)", self.Pl2_m2(l3, -1), m_nu, -1) 
        self._check("m^2_nu(lambda)", self.Pl2_m2(l4, -1), m_nu, -1) 

    def proof_affine_relation(self):
        opx, opm = self.Om / self.Op, self.Op / self.Om 
        z2p, z2m = self.pls.Z2, self.mns.Z2
        vnu = self.F_frame(self.nu)

        ln = linear_t(self)
        ppmm = ln.eta(ln.alpha(self.nu.mass, +1, +1), ln.alpha(self.nu.mass, +1, -1))
        mpmp = ln.eta(ln.alpha(self.nu.mass, -1, +1), ln.alpha(self.nu.mass, +1, -1))


#        print(ln.P_nu(tau, eta, chi, +1, +1))
        print(ppmm, mpmp)
        exit()
        print(eta)
#        x = ln.eta(ln.alpha(self.nu.mass, -1, +1), ln.alpha(self.nu.mass, +1, +1))
#        print(x)


        exit()
        app, apm = ln.alpha(self.nu.mass, +1, +1), ln.alpha(self.nu.mass, +1, -1)
        amp, amm = ln.alpha(self.nu.mass, -1, +1), ln.alpha(self.nu.mass, -1, -1)
            
        print(app, apm)
        print(amp, amm)


        exit()

        self._check("[ a^2[+] - a^2[-] ]^2", opx ** 2, (app ** 2 - apm ** 2) ** 2)




        print( (amm * amp), -( (apm * amm) * 1 / (self.Op * self.Om)**2 ) ** -1 )
    
        print(ln.eta(ln.aM(0.01), ln.aP(0.01), (app ** 2 - apm ** 2) ** 2, "|", opx ** 2, opm ** 2 ))
#        print( (apm ** 2 + amm ** 2) ** 2, "|", opx ** 2, opm ** 2 )

        


        exit()




        self._check("l[+]l[-] = (O[+] / O[-])^2"   , ln.lF(self.nu.mass, +1) * ln.lF(self.nu.mass, -1), (self.Om / self.Op)**2)
        self._check("a[+]^2 - a[-]^2 = O[-] / O[+]", ln.aP(0.01)**2 - ln.aM(0.01) ** 2                , self.Om / self.Op)



        exit()



        self._check("eta(m_nu) = 0.01"             , 0.01                                             , ln.eta(ln.aM(0.01), ln.aP(0.01)))


        exit()
        self._check("Nu-Mass", self.nu.mass, ln.m_nu(eta), eta)

        eta = ln.eta(ln.alpha(self.nu.mass, -1), ln.alpha(self.nu.mass, +1))
        chi = ln.t_chi(vnu, +1)
        tau = ln.t_tau(vnu, +1)
        
        print(ln.lF(self.nu.mass, +1), ln.lK(self.nu.mass, -1))

        print(tau)
        print(vnu)
        exit()

