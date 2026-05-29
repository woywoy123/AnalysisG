from atomics import *
import numpy as np

class sols:
    def __init__(self, z2, Sx, kappa, s):
        Sy = Sx / kappa
        self.Sx = Sx; self.Sy = Sy
        self.x1 = x1(Sx, Sy, z2.w, z2.o)
        self.y1 = y1(Sx, Sy, z2.w, z2.o)
        self.z2 = z2.fx(Sx, Sy)
        self.kappa = kappa
        self.sign = s

    def __str__(self):
        x  = " sx: " + str(self.Sx) + " sy: " + str(self.Sy) 
        x += " x1: " + str(self.x1) + " y1: " + str(self.y1) 
        x += " z2: " + str(self.z2) + " kappa: " + str(self.kappa)
        x += " sign: " + self.sign
        return x

class Z2:
    def __init__(self, w, o2, data, s):
        self.s = s

        a = (1 / o2 - 1)
        b = 2 * w / o2 
        c = (w ** 2 / o2 - 1)
        d = 2 * data.lep.p
        e = data.lep.mass ** 2 - data.m_nu ** 2
        self.coef = [a, b, c, d, e]

        self.w, self.o2, self.o = w, o2, o2 ** 0.5
        self.lep = data.lep
        self.m_nu = data.m_nu

    def fx(self, Sx, Sy):
        ic = iter(self.coef)
        return sum([next(ic) * i for i in [Sx ** 2, Sx * Sy, Sy ** 2, Sx, 1]])

    def points(self, kappa, solve = True):
        a = ((self.lep.b ** 2 - self.w**2) + 2 * self.w * kappa - (1 - self.lep.b ** 2) * kappa**2) * 1 / self.o2
        b = 2 * self.lep.b * self.lep.e
        c = (self.lep.mass ** 2 - self.m_nu ** 2)
        if not solve: return [a, b, c]
        dsc = complex(b ** 2 - 4 * a * c)**0.5
        SxP, SxM = (-b + dsc.real)/(2 * a), (-b - dsc.real)/(2 * a)
        return [sols(self, SxP, kappa, self.s + "+"), sols(self, SxM, kappa, self.s + "-")]

    def diff(self, kappa):
        a = - self.lep.b * self.lep.e * self.o2 
        b = self.lep.b ** 2 - self.w ** 2 + 2 * self.w * kappa - (1 - self.lep.b ** 2) * kappa ** 2 

        dbl = (self.lep.b * self.lep.e) ** 2 * self.o2
        dbl = dbl / ((1 - self.lep.b **2) * kappa ** 2 - 2 * self.w * kappa - self.lep.b ** 2 + self.w**2)
        m_nu = dbl**0.5

        Sx, Sy, z2 = a / b, (a / b) * kappa, self.fx(a / b, (a / b) * kappa)
        return Sx, Sy, x1(Sx, Sy, self.w, self.o), y1(Sx, Sy, self.w, self.o), z2, H_tilde(z2 ** 0.5, Sx, Sy, self)


class dG2:
    def __init__(self, data, truth = None):
        self.GP = (data.wp + data.wm) / data.o2p
        self.GM = (data.wp - data.wm) / data.o2m

        self.a = (data.wm - data.wp) * (data.wm + data.wp) / (data.o2p * data.o2m)
        self.b = (data.wp * data.o2m - data.wm * data.o2p) / (data.o2p * data.o2m)
        self.c = - (1 - data.lep.b ** 2) * self.a 
        self.coef = [self.a, 2 * self.b, self.c]
        self.quad = np.array([[self.a, self.b], [self.b, self.c]])
        self.qdet = - (data.wp - data.wm)**2 / (data.o2p * data.o2m)
    
        # ------ roots ----- # 
        self.dp = ((data.o2p ** 0.5 - data.o2m ** 0.5) ** 2 - (data.wp + data.wm) ** 2)/(2 * (data.wp + data.wm))
        self.dm = ((data.o2p ** 0.5 + data.o2m ** 0.5) ** 2 - (data.wp + data.wm) ** 2)/(2 * (data.wp + data.wm))
     
        self.ppsi = math.atan(self.dp)
        self.mpsi = math.atan(self.dm)
        self.psi  = (self.ppsi + self.mpsi) / 2

        f = self.GP * self.GM / (2 * math.cos(self.ppsi) * math.cos(self.mpsi)) 
        self.lp = - f * (math.cos(self.ppsi - self.mpsi) - 1)
        self.lm = - f * (math.cos(self.ppsi - self.mpsi) + 1)
        print(self.lp, self.lm)

        sxmm, sxpp = data._Z2m.diff(1 / self.dm), data._Z2p.diff(1 / self.dp)
        sxmp, sxpm = data._Z2m.diff(1 / self.dp), data._Z2p.diff(1 / self.dm)

        mx11, my11 = sxmm[2], sxmm[3]
        mx12, my12 = sxpm[2], sxpm[3]
        gr_m = (my12 - my11)/(mx12 - mx11)
        bm = my12 - gr_m * mx12

        px11, py11 = sxpp[2], sxpp[3]
        px12, py12 = sxmp[2], sxmp[3]
        gr_p = (py12 - py11)/(px12 - px11)
        bp = py12 - gr_p * px12

        x0 = (bm - bp)/(gr_p - gr_m)
        y0 = gr_m * x0 - bm 

        elp = [sxmm[-1], sxpp[-1], sxmp[-1], sxpm[-1], truth.H_tilde] 

        print("!-> branches intersect", y0, x0)
        print("!-> m(kappa^+-)", gr_m, gr_p)
        print("!-> origin", bm, bp)

        exit()
        self.eigv = np.array([
            [math.cos(self.psi),  - math.sin(self.psi) ], 
            [math.sin(self.psi),    math.cos(self.psi) ]
        ])
        self.eiga = np.array([
            [math.sin(self.ppsi), math.sin(self.mpsi)], 
            [math.cos(self.ppsi), math.cos(self.mpsi)]
        ])



#        print("->", data.lep.b, data.bqrk.b, data.theta)
#        print(math.cos(self.ppsi - self.mpsi) / (math.cos(self.ppsi) * math.cos(self.mpsi)), data.lep.b**2)
#        print(self.eiga)
#        print(self.eigv)
        #self.alpha  = (1 - (data.lep.b * math.sin(self.psi))**2)/(1 - (data.lep.b * math.cos(self.psi))**2)
        #self.alpha  = math.atan(self.alpha ** 0.5)
        #print(math.tan(self.psi - self.alpha) * math.tan(self.psi + self.alpha), - 1 / ( 1 - data.lep.b**2))
        #print(- 1 / math.tan(self.psi - self.alpha), - 1 / math.tan(self.psi + self.alpha))
        #print(self.dp, self.dm)
        #print(math.atan(data.wp), math.atan(data.wm), - self.psi, self.alpha, self.mpsi, - self.ppsi)
        #print(math.atan(data.wp) - self.psi, math.atan(data.wm) + self.alpha)

        #return 

        #sol  = [data._Z2m.points(self.dm), data._Z2p.points(self.dp)]
        #sol += [data._Z2m.points(self.dp), data._Z2p.points(self.dm)]
        #dic = {"++" : [], "--" : [], "+-" : [], "-+" : []}
        #for i in sum(sol, []): dic[i.sign].append(i)
#            print(mT(data.m_nu, data.lep, data.bqrk, data.sth, data.cth, i.Sx, i.Sy), mW(data.m_nu, data.lep, i.Sx))
        #mm, pp = data._Z2m.points(self.dm, False), data._Z2p.points(self.dp, False)
        #mp, pm = data._Z2m.points(self.dp, False), data._Z2p.points(self.dm, False)
        
        #print("->", data.m_nu)
        for i in elp:
            print(i)
            print("____")

        #print("--", [sxmm[i+2] for i in range(3)])
        #print("++", [sxpp[i+2] for i in range(3)])
        #print("+-", [sxpm[i+2] for i in range(3)])
        #print("-+", [sxmp[i+2] for i in range(3)])
        #elp = [sxmm[-1], sxpp[-1], sxmp[-1], sxpm[-1]]
        from figures import plot_ellipses

        self.cols  = iter(["black", "navy", "cyan", "green", "blue", "navy", "cyan", "green"])
        self.style = iter(["--", "-.", ":", "solid", "-", "-", "-", "-"])
        plot_ellipses([ [ i, next(self.cols), next(self.style) ] for i in elp], None, truth.solution["htilde"], np.array([x0, y0, 0]))
        return
        #import matplotlib.pyplot as plt
        #pp = np.array([i.z2.real for i in dic["++"]])
        #pm = np.array([i.z2.real for i in dic["+-"]])
        #mp = np.array([i.z2.real for i in dic["-+"]])
        #mm = np.array([i.z2.real for i in dic["--"]])

        #def quads(sx, coef): return coef[0] * sx ** 2 + coef[1] * sx + coef[0]
        #sx = np.linspace(-200, 200, 10000)
        #mm, pp = quads(sx, mm), quads(sx, pp)
        #mp, pm = quads(sx, mp), quads(sx, pm)

        #fig1, ax1 = plt.subplots(1, 1, figsize=(15, 5))
        #ax1.plot(sx, mm, color="blue"  , linestyle = "-.")
        #ax1.plot(sx, pp, color="black" , linestyle = "-")
        #ax1.plot(sx, mp, color="red"   , linestyle = "-.")
        #ax1.plot(sx, pm, color="orange", linestyle = "-")

        #ax1.set_xlabel('x (GeV)')
        #ax1.set_ylabel('y (GeV)')
        #ax1.grid(True, alpha=0.3)
        #plt.tight_layout()
        #plt.show()



        exit()





















        print(pp, mm)
        print(pm, mp)

        return 
        exit()

        rot = [i.z2.real for i in sum(sol, [])]

        print(np.linalg.eig(np.array(rot).reshape(2, 2)))
        return 


        #for i in sol:
        #    print(mT(data.m_nu, data.lep, data.bqrk, data.sth, data.cth, i.Sx, i.Sy))
        #exit()
        # ------ eigenvalues ------ #
        sol += [data._Z2p.points( (self.dm + self.dp) / 2 ), data._Z2m.points( (self.dm + self.dp) / 2 )]
        sol  = sum(sol, [])

 
    def fx(self, Sx, Sy):
        ic = iter(self.coef)
        return sum([next(ic) * i for i in [Sx ** 2, Sx * Sy, Sy ** 2]])

    def fc(self, Sx, Sy):
        return - self.Gpm * (Sx - self.dP * Sy) * (Sx - self.dM * Sy)


class variables:
    def __init__(self, lep, bqrk, nu = None, truth = None):
        self.cth = costheta(lep, bqrk)
        self.sth = (1 - self.cth ** 2)**0.5
        self.tth = self.sth / self.cth
        self.theta = math.acos(self.cth)

        self.rho = lep.b / bqrk.b 

        self.m_nu = nu.mass if nu is not None else 0
        self.m_W  = 0
        self.m_T  = 0

        self.lep  = lep
        self.bqrk = bqrk

        self.wp  =   (1 / self.sth) * (self.rho - self.cth)
        self.wm  = - (1 / self.sth) * (self.rho + self.cth)

        # -------- alternative wm and wp ------- # 
        self._wp =  0.5 * ((self.rho + 1)*math.tan(self.theta / 2) + (self.rho - 1)*(1 / math.tan(self.theta / 2)))
        self._wm = -0.5 * ((self.rho - 1)*math.tan(self.theta / 2) + (self.rho + 1)*(1 / math.tan(self.theta / 2)))

        self.o2m = self.wm ** 2 + 1 - lep.b ** 2
        self.o2p = self.wp ** 2 + 1 - lep.b ** 2
        
        self._Z2p = Z2(self.wp, self.o2p, self, "+")
        self._Z2m = Z2(self.wm, self.o2m, self, "-")
        self._dG2 = dG2(self, truth)

