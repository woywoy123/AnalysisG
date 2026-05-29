from conuix.visuals.visual import PlotQuadrics, packet
from conuix.types.base import structs_t
from conuix.visuals.geometry import *
from conuix.types.structs import *
import numpy as np

class figures:
    def __init__(self): pass
       
    def make_4x4(self, ix):
        x = np.vstack((ix, np.array([[0, 0, 0]])))
        return np.hstack((x, np.array([[0], [0], [0], [1]])))

    def surfaces(self):
        sx, sy = self.pls.Sx, self.pls.Sy

        plx = []
#        plx += [TwoSheet("Q^2[+]", "purple", 1000000, self.MQ(+1) - np.diag([0, 0, 0, self.nu.mass ** 2]))]
#        plx += [TwoSheet("Q^2[-]", "red"   , 1000000, self.MQ(-1) - np.diag([0, 0, 0, self.nu.mass ** 2]))]

#        plx += [TwoSheetS0("Q^2[+]'", "blue", 1000000, self.MQ(+1) - np.diag([0, 0, 0, self.nu.mass ** 2]), self, +1)]
#        plx += [TwoSheetS0("Q^2[-]'", "red" , 1000000, self.MQ(-1) - np.diag([0, 0, 0, self.nu.mass ** 2]), self, -1)]

        print(self.MQ(+1))
        plx += [TwoSheetS0R("Q^2[+]", "purple", 1000000, - np.diag([0, 0, 0, self.nu.mass ** 2]), self, +1)]
        plx += [TwoSheetS0R("Q^2[-]", "red"   , 1000000, - np.diag([0, 0, 0, self.nu.mass ** 2]), self, -1)]

#        plx += [TwoSheetT("Q[+] - Truth", "purple"   , 1000000, self.MQ(+1) - np.diag([0, 0, 0, self.nu.mass ** 2]), [-sx, -sy, 0])] #sx, sy, self.pls.Z2])]
#        plx += [TwoSheetT("Q[-] - Truth", "red"      , 1000000, self.MQ(-1) - np.diag([0, 0, 0, self.nu.mass ** 2]), [-sx, -sy, 0])] #sx, sy, self.mns.Z2])]

#        plx += [Ellipse("Ellipse[w+] - Truth", "blue", 1000000, self.pls.H_tilde)]
#        plx += [Ellipse("Ellipse[w-] - Truth", "red" , 1000000, self.mns.H_tilde)]

#        plx += [ParticleE("b-quark[w+]", "green"  , 2000000, self.BQ(sx, sy, -(self.pls.Z2 + self.nu.mass ** 2) ))]
#        plx += [ParticleE("lepton[w+]" , "yellow" , 2000000, self.LQ(sx, sy, -(self.pls.Z2 + self.nu.mass ** 2) ))]

        PlotQuadrics(plx); plx = []
        exit()

        return 
        nu = self.F_frame(self.nu)

        sx, sy, z2 = self.pls.Sx, self.pls.Sy, self.pls.Z2
        chi = np.linspace(0, 2 * np.pi, 10000000)
        nup = self.vec_nu(sx, sy, z2 ** 0.5, chi, +1).T
        chi = chi[np.argmin( ((nup - nu) ** 2 ).sum(-1) ) ]
        nuP = self.vec_nu(sx, sy, z2 ** 0.5, chi, +1)
        plx = []


#        e, v = np.linalg.eig( (self.MQ(+1) - self.MQ(-1)).dot((self.MQ(-1) + self.MQ(+1)).T) )
        mq, mp = self.MQ(+1) - np.diag([1, 1, 1, self.nu.mass ** 2]), self.MQ(-1) - np.diag([1, 1, 1, self.nu.mass ** 2])
#        print( 2 * (self.m_mu * (self.wp - self.wm) / (self.Op * self.Om * self.b_mu) ) ** 2 )
#        mq = sH.T.dot(mq).dot(sH) # - mq)
        x = mp.dot(np.linalg.inv(mq).T) - mq.dot(np.linalg.inv(mp).T)
        print(np.linalg.eig(x)[0])
        print(x)
        exit()
#        plx += [ParticleE("xb-quark", "green" , 2000000, self.BQ(sx, sy, -(self.pls.Z2 + self.nu.mass ** 2) ))]
#        plx += [ParticleE("xlepton", "yellow" , 2000000, self.LQ(sx, sy, -(self.pls.Z2 + self.nu.mass ** 2) ))]
#
        #plx += [ParticleQ("b-quark[+]"  , "green"  , 1000000, self.MQ(+1), self.BQ)]
        #plx += [ParticleQ("b-quark[-]"  , "yellow" , 1000000, self.MQ(-1), self.BQ)]
        #plx += [ParticleQ("lepton[+]"   , "purple" , 1000000, self.MQ(+1), self.LQ)]
        #plx += [ParticleQ("lepton[-]"   , "red"    , 1000000, self.MQ(-1), self.LQ)]

#        l = 5000
#        pl = packet(domain = [[-l, l] for i in range(3)])
#        pl.add_truth(self.pls.H_tilde, pts = nuP)
#        pl.add_ellipse(lx(10, 10, +1), "l+")
#        pl.add_ellipse(lx(10, 10, -1), "l-")
#        pl.compile2DProj()


        
        z2pp, z2pm = lb.Z2P(lb.lxp, lb.lyp, m_nu, 0, 0), lb.Z2P(lb.lxp, lb.lym, m_nu, 0, 0)
        z2mp, z2mm = lb.Z2M(lb.lxm, lb.lyp, m_nu, 0, 0), lb.Z2M(lb.lxm, lb.lym, m_nu, 0, 0)

        






        exit()

        r = (self.m_mu / self.b_mu)**2 * (self.wp - self.wm) ** 2 * self.Gp * self.Gm
        print(self.Sy0(+1) - self.Sy0(-1), - (self.p_mu / self.b_mu**2) * (self.wp - self.wm))
        exit()

        nu = self.F_frame(self.nu)

        chi = np.linspace(0, 2 * np.pi, 10000000)
        nup = self.vec_nu(sx, sy, z2p ** 0.5, chi, +1).T
        chi = chi[np.argmin( ((nup - nu) ** 2 ).sum(-1) ) ]
        nuP = self.vec_nu(sx, sy, z2p ** 0.5, chi, +1)

        from conuix.visuals.visual import packet
        print(nu)
        for i in np.linspace(-100000000, 10000, 1000000):
            z2pp, z2pm = lb.Z2P(lb.lxp, lb.lyp, m_nu, 0, i), lb.Z2P(lb.lxp, lb.lym, m_nu, 0, i)
            z2mp, z2mm = lb.Z2M(lb.lxm, lb.lyp, m_nu, 0, i), lb.Z2M(lb.lxm, lb.lym, m_nu, 0, i)
            v1 = np.array([lb.lxp, lb.lyp, z2pp])
            v2 = np.array([lb.lxp, lb.lym, z2mp])
            v3 = np.array([lb.lxm, lb.lyp, z2pm])
            v4 = np.array([lb.lxm, lb.lym, z2mm])
            xp = np.array([[z2pp, z2pm], [z2mp, z2mm]])

            l1, m1 = v1 - v2, np.cross(v1, v2)
            l2, m2 = v1 - v3, np.cross(v1, v3)
            l3, m3 = v1 - v4, np.cross(v1, v4)

            l4, m4 = v2 - v3, np.cross(v2, v3)
            l5, m5 = v2 - v4, np.cross(v2, v4)
            l6, m6 = v3 - v4, np.cross(v1, v4)
            print(sum([k.dot(j).item() for k, j in zip([l1, l2, l3, l4, l5, l6], [m1, m2, m3, m4, m5, m6])]), i, xp.flatten())


            continue

            pl = packet(domain = [[-10000, 10000] for i in range(3)])
            pl.add_truth(self.pls.H_tilde, pts = nuP)
            pl.add_ellipse(lb.Htilde(lb.lxp, lb.lyp, 0, +1, i, m_nu), "l++")
            pl.add_ellipse(lb.Htilde(lb.lxp, lb.lym, 0, +1, i, m_nu), "l+-")
#            pl.add_ellipse(lb.Htilde(lb.lxm, lb.lyp, 0, -1, i, m_nu), "l-+")
#            pl.add_ellipse(lb.Htilde(lb.lxm, lb.lym, 0, -1, i, m_nu), "l--")

            pl.compile2DProj()




            #print(xp, np.cross(xp[0], xp[1]))
        exit()








    

        exit()




















#    def Transformation(self): 
#
#        plx = []
#        sx, sy, z2 = self.pls.Sx, self.pls.Sy, self.pls.Z2
#        rt = self.make_4x4(self.R_T)
#        self.nx = self.F_frame(self.nu)
#    
#        # ------------- Raw Parameterization ---------- #
#        # --------> important!! This is after rotation <------ !!!
#        Hmn, Hmp = self.mns.H, self.pls.H
#        plx += [Point("Neutrino", "Cyan", 10000, self.nx)]
#        plx += [TwoSheet("Q[-] - Truth", "purple", 10000, self.make_4x4(Hmn))]
#        plx += [TwoSheet("Q[+] - Truth", "green" , 10000, self.make_4x4(Hmp))]
#
##        plx += [TwoSheet("Q[+] - Param", "red" , 10000, rt.dot(bpl.H(self.nu.mass)))]
##        plx += [TwoSheet("Q[-] - Param", "blue", 10000, rt.dot(bmn.H(self.nu.mass)))]
#        PlotQuadrics(plx); plx = []
#      
#        # SHIFT THE SURFACE TO THE CENTER
#        # ------------- S' = S - S0 -------------- #
##        plx += [TwoSheetS0("Q'[+] - Param", "red" , 1000000, bpl.H(self.nu.mass), bpl)]
##        plx += [TwoSheetS0("Q'[-] - Param", "blue", 1000000, bmn.H(self.nu.mass), bmn)]
##        plx += [Point("Neutrino", "Cyan", 1000000, self.nx)]
##        PlotQuadrics(plx); plx = []
#        
#        # SHIFT THE SURFACE TO THE CENTER + rotate 
#        # ------------- S' = S - S0 -------------- #
##        plx += [TwoSheetS0R("Q''[+] - Param", "red" , 1000000, bpl.H(self.nu.mass), bpl)]
##        plx += [TwoSheetS0R("Q''[-] - Param", "blue", 1000000, bmn.H(self.nu.mass), bmn)]
##        plx += [Point("Neutrino", "Cyan", 1000000, self.nx)]
##        PlotQuadrics(plx); plx = []
#        return 
#        exit()
#
#
#        # LAMBDA PARAMETERIZATION
#        # ---------------------------------- #
#       # pl = pencil_L1(self.bq, self.lp)
#        #plx += [TwoSheetLX("Z^2[+, +]", "red" , 1000000, bpl.H(self.nu.mass), pl, +1)]
#        #plx += [TwoSheetLX("Z^2[-, -]", "blue", 1000000, bmn.H(self.nu.mass), pl, -1)]
#        #plx += [TwoSheetLX("Z^2[+, -]", "green" , 1000000, bpl.H(self.nu.mass), pl, -1)]
#        #plx += [TwoSheetLX("Z^2[-, +]", "purple", 1000000, bmn.H(self.nu.mass), pl, +1)]
#        #plx += [Point("Neutrino", "Cyan", 1000000, self.nx)]
#        #PlotQuadrics(plx); plx = []
#
#        plx += [TwoSheetL1("Z^2[+, +]", "red" , 1000000, bpl.H(self.nu.mass), pl, +1)]
#        plx += [TwoSheetL1("Z^2[-, -]", "blue", 1000000, bmn.H(self.nu.mass), pl, -1)]
##        plx += [TwoSheetL1("Z^2[+, -]", "green" , 1000000, bpl.H(self.nu.mass), pl, -1)]
##        plx += [TwoSheetL1("Z^2[-, +]", "purple", 1000000, bmn.H(self.nu.mass), pl, -1)]
#        plx += [Point("Neutrino", "Cyan", 1000000, self.nx)]
#        PlotQuadrics(plx); plx = []
#
#
#
#
#
#
#
#
#
#
#
#        exit() 
#
#        l0pp, l0mp = self.to_Lp0(0, +1), self.to_Lm0(0, +1)
#        l0pm, l0mm = self.to_Lp0(0, -1), self.to_Lm0(0, -1)
#
#        x = [
#            [l0pp, l0mp, self.Z2L(l0pp, l0mp, m_nu, +1)], 
#            [l0pp, l0mp, self.Z2L(l0pp, l0mp, m_nu, -1)], 
#            
#            [l0pm, l0mp, self.Z2L(l0pm, l0mp, m_nu, +1)],
#            [l0pm, l0mp, self.Z2L(l0pm, l0mp, m_nu, -1)],
#            
#            [l0pm, l0mm, self.Z2L(l0pm, l0mm, m_nu, +1)],
#            [l0pm, l0mm, self.Z2L(l0pm, l0mm, m_nu, -1)]
#        ]
#
#        dk = [] 
#        t = np.array(x)
#        for i in range(len(t)): 
#            for j in range(len(t)):
#                if i >= j: continue
#                l = t[i] - t[j]
#                m = np.cross(t[i], t[j])
#                # Grassmann - Plucker relation:
#                # D dot M = 0 
#                if abs(d.dot(m)) > 0: continue
#                s = 1 / sum(d**2) ** 0.5
#                dk.append([d, t[i], t[j], d * s, m * s])
#                print(d * s)
#                print(m * s)
#                print("____")
# 
#
#        exit()
# 
#        import matplotlib.pyplot as plt
#        t = np.linspace(0, 2*np.pi, 100)
#        ellipse_x = x[1][-1] * np.cos(t)
#        ellipse_y = x[1][-1] * np.sin(t)
#        ellipse_z = x[1][-1] * np.zeros_like(t)
#       
#        z = z2p**0.5 
#        ht = self.htilde(sx, sy, z, +1)
#
#        line_length = 2
#        L_plus_x = np.linspace(-line_length, line_length, 10)
#        L_plus_y = L_plus_x * 0.5
#        L_plus_z = L_plus_x * 0.2
#        
#        L_minus_x = np.linspace(-line_length, line_length, 10)
#        L_minus_y = -L_minus_x * 0.5
#        L_minus_z = -L_minus_x * 0.2
#        shift_vector = dk[0][0]
#        
#        # Translate the ellipse
#        trans_ellipse_x = ellipse_x + shift_vector[0]
#        trans_ellipse_y = ellipse_y + shift_vector[1]
#        trans_ellipse_z = ellipse_z + shift_vector[2]
#        
#        # Translate the basis lines
#        trans_L_plus_x = L_plus_x + shift_vector[0]
#        trans_L_plus_y = L_plus_y + shift_vector[1]
#        trans_L_plus_z = L_plus_z + shift_vector[2]
#        
#        trans_L_minus_x = L_minus_x + shift_vector[0]
#        trans_L_minus_y = L_minus_y + shift_vector[1]
#        trans_L_minus_z = L_minus_z + shift_vector[2]
#        
#        # 3. Plotting with Matplotlib
#        fig = plt.figure(figsize=(10, 8))
#        ax = fig.add_subplot(111, projection='3d')
#        
#        # Plot Origin Geometry (The pristine, factorable intersection)
#        ax.plot(ellipse_x, ellipse_y, ellipse_z, label='Origin Phase Ellipse', color='blue', linewidth=2)
#        ax.plot(L_plus_x, L_plus_y, L_plus_z, label='Origin L+', color='cornflowerblue', linestyle='-.')
#        ax.plot(L_minus_x, L_minus_y, L_minus_z, label='Origin L-', color='cornflowerblue', linestyle='-.')
#        
#        # Plot Translated Physical Geometry (The true lab frame kinematics)
#        ax.plot(trans_ellipse_x, trans_ellipse_y, trans_ellipse_z, label='Physical Phase Ellipse', color='red', linewidth=2)
#        ax.plot(trans_L_plus_x, trans_L_plus_y, trans_L_plus_z, label='Physical L+', color='salmon', linestyle='-.')
#        ax.plot(trans_L_minus_x, trans_L_minus_y, trans_L_minus_z, label='Physical L-', color='salmon', linestyle='-.')
#        
#        # Plot the Translation Vector (The Affine Push)
#        ax.plot([0, shift_vector[0]], [0, shift_vector[1]], [0, shift_vector[2]], 
#                label='Affine Shift Vector (M_epsilon col 5)', color='green', linewidth=3, linestyle='--')
#        # Add a marker at the end of the arrow
#        ax.scatter(*shift_vector, color='green', s=50, zorder=5)
#        ax.scatter(l0pm, l0mm, x[-2][2]*0, color='black', s=50, label='Origin (0,0,0)', zorder=5)
#        
#        # Formatting
#        ax.set_title('Affine Translation of the Kinematic Phase Space', fontsize=14, pad=20)
#        ax.set_xlabel('X Momentum')
#        ax.set_ylabel('Y Momentum')
#        ax.set_zlabel('Z Momentum')
#        
#        
#        max_range = np.array([
#            ellipse_x.max()-ellipse_x.min(), 
#            ellipse_y.max()-ellipse_y.min(), 
#            trans_ellipse_z.max()-ellipse_z.min()
#        ]).max() / 2.0
#        
#        mid_x = (ellipse_x.max()+ellipse_x.min()) * 0.5 + shift_vector[0]/2
#        mid_y = (ellipse_y.max()+ellipse_y.min()) * 0.5 + shift_vector[1]/2
#        mid_z = (trans_ellipse_z.max()+ellipse_z.min()) * 0.5
#        
#        ax.set_xlim(mid_x - max_range, mid_x + max_range)
#        ax.set_ylim(mid_y - max_range, mid_y + max_range)
#        ax.set_zlim(mid_z - max_range, mid_z + max_range)
#        
#        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
#        plt.tight_layout()
#        plt.show()
#   
#
