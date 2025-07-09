import numpy as np
import math
from math import sin, cos, sqrt

def Z2(Eb, Em, Pm, th, Bb, Bm, mb, mm, mT, mW):
    x0p = -(mT**2 - mW**2 - mb**2)/(2*Eb)
    x0  = -(mW**2 - mm**2        )/(2*Em)
    Sx  = (x0 * Bm - Pm*(1-Bm**2))/Bm
    Sy  = (x0p/Bb - cos(th)*Sx)/sin(th)
    w   = (Bm/Bb - cos(th))/sin(th)
    om2 = (w**2 + 1 - Bm**2)
    
    eps2 = (mW**2)*(1-Bm**2) 
    x1 = Sx - (Sx + w * Sy)/om2
    y1 = Sy - (Sx + w * Sy)*(w/om2)
    return x1**2 * om2 - (Sy - w*Sx)**2 - (mW**2 - x0**2 - eps2)

def get_mW(Eb, Em, theta, Bb, Bm, mb, mm, mT):
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    w = (Bm / Bb - cos_theta) / sin_theta
    om2 = w**2 + 1 - Bm**2
    
    E0 = mm**2 / (2 * Em)
    E1 = -1 / (2 * Em)
    
    P1 = 1 / (2 * Eb)
    P0 = (mb**2 - mT**2) / (2 * Eb)
    Sx = E0 - mm**2 / Em 
    
    Sy0 = (P0 / Bb - cos_theta * Sx) / sin_theta
    Sy1 = (P1 / Bb - cos_theta * E1) / sin_theta
    
    X0 = Sx * (1 - 1/om2) - (w * Sy0) / om2
    X1 = E1 * (1 - 1/om2) - (w * Sy1) / om2
    
    D0 = Sy0 - w * Sx
    D1 = Sy1 - w * E1
    
    # Quadratic coefficients for Z2 = A*v² + B*v + C
    A_val = om2 * X1**2 - D1**2 + E1**2
    B_val = 2 * (om2 * X0 * X1 - D0 * D1) + 2 * E0 * E1 - Bm**2
   
    v_crit, v_infl = 0, 0
    v_crit = -B_val / (2 * (A_val if A_val != 0 else 1))
    v_infl = -B_val / (6 * (A_val if A_val != 0 else 1))
    if v_crit >= 0: v_crit = math.sqrt(v_crit)
    if v_infl >= 0: v_infl = math.sqrt(v_infl)
    return v_crit, v_infl


def get_mT2(Eb, Em, th, Bb, Bm, mb, mm, mW1, mW2, mT1):
    # Common terms
    sin_th = math.sin(th)
    cos_th = math.cos(th)
    w = (Bm / Bb - cos_th) / sin_th
    Omega = w**2 + 1 - Bm**2

    x0   = -(mW1**2 - mm) / (2 * Em)
    Sx   = x0 - Em * (1 - Bm**2) 
    x0p  = -(mT1**2 - mW1**2 - mb) / (2 * Eb)
    Sy   = (x0p / Bb - cos_th * Sx) / sin_th
    x1   = Sx - (Sx + w * Sy) / Omega
    Z2y  = (x1**2 * Omega) - (Sy - w * Sx)**2 - (mW1**2 - x0**2 - mW1**2 * (1 - Bm**2))

    x0_   = -(mW2**2 - mm) / (2 * Em)
    Sx_   = x0_ - Em * (1 - Bm**2)
    eps2_ = mW2**2 * (1 - Bm**2)
    cons  = mW2**2 - x0_**2 - eps2_

    A_sy = -1 / (2 * Eb * Bb * sin_th)
    B_sy = ((mW2**2 + mb) / (2 * Eb * Bb) - cos_th * Sx_) / sin_th

    A_x1 = - (w * A_sy) / Omega
    B_x1 = ((Omega - 1) * Sx_ - w * B_sy) / Omega
    A    = Omega * A_x1**2 - A_sy**2
    B    = 2 * Omega * A_x1 * B_x1 - 2 * A_sy * (B_sy - w * Sx_)
    C    = Omega * B_x1**2 - (B_sy - w * Sx_)**2 - cons- Z2y
    discriminant = B**2 - 4 * A * C

    root1 = (-B + math.sqrt(discriminant)) / (2 * A)
    root2 = (-B - math.sqrt(discriminant)) / (2 * A)
   
    root1 = math.sqrt(root1) if root1 >= 0 else 0
    root2 = math.sqrt(root2) if root2 >= 0 else 0
    return min(root1, root2)


def grid(Eb, Em, Pm, th, Bb, Bm, mb, mm, mn, mT, mW, itx, ity):

    mW1 = np.ones(itx) #linspace(0.01, 2.5, itx)
    mT1 = np.linspace(0.1, 1.5, itx) # np.ones(itx)

    #mT1, mW1 = np.meshgrid(u, u)
    mT1 = mT*mT1.reshape(-1)
    mW1 = mW*mW1.reshape(-1)

    nu  = Z2(Eb, Em, Pm, th, Bb, Bm, mb, mm, mT1, mW1).reshape(-1)
    msk = nu != 1000000000
    mT1 = mT1[msk.reshape(-1)]*0.001
    mW1 = mW1[msk.reshape(-1)]*0.001
    lx  = nu.size
    o = np.concatenate((nu.reshape(lx, 1), mT1.reshape(lx, 1), mW1.reshape(lx, 1)), -1)
    return o


import matplotlib.pyplot as plt
mw1 = 76.76539525803744 
mw2 = 93.56876906646316 
mt1 = 142.70374747939857 
mt2 = 164.54881452848292                                                                                                                                   

# -------- nu1 -------- #
Eb1 =  83191.328125
Em1 =  49295.99609375
Pm1 =  49295.88272804186
th1 =  0.9336250845615144
Bb1 =  0.9964928002447768
Bm1 =  0.9999977003059656
mb1 =  6961.329377134628
mm1 =  105.7210393415889

mT1 = 172.62*1000
mW1 = 80.385*1000
mN1 = 0


# -------- nu2 -------- #
Eb2 =  225794.953125
Em2 =  146976.546875
Pm2 =  146976.5145921903
th2 =  0.6217613287483583
Bb2 =  0.9961177745598501
Bm2 =  0.9999997803540063
mb2 =  19876.85579943639
mm2 =  97.41473576894865

mT2 = 172.62*1000
mW2 = 80.385*1000
mN2 = 0

import random
number_of_colors = 120
col = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]

fig = plt.figure()
ax = plt.axes() #projection='3d') 
for i in range(len(col)):
    fx1 = grid(Eb1, Em1, Pm1, th1, Bb1, Bm1, mb1, mm1, mN1, mT1, mW1, 100, 100)
    fx2 = grid(Eb2, Em2, Pm2, th2, Bb2, Bm2, mb2, mm2, mN2, mT2, mW2, 100, 100)

    mw1 = get_mW(Eb1, Em1, th1, Bb1, Bm1, mb1, mm1, mT1)
    mw2 = get_mW(Eb2, Em2, th2, Bb2, Bm2, mb2, mm2, mT2)

    Z1x = Z2(Eb1, Em1, Pm1, th1, Bb1, Bm1, mb1, mm1, mT1, mW1)
    Z2x = Z2(Eb2, Em2, Pm2, th2, Bb2, Bm2, mb2, mm2, mT2, mW2)

    Z1y = Z2(Eb1, Em1, Pm1, th1, Bb1, Bm1, mb1, mm1, mT1, mw1[0])
    Z2y = Z2(Eb2, Em2, Pm2, th2, Bb2, Bm2, mb2, mm2, mT2, mw2[0])

    _mT1 = get_mT2(Eb1, Em1, th1, Bb1, Bm1, mb1, mm1, mW1, mw1[0], mT1)
    _mT2 = get_mT2(Eb2, Em2, th2, Bb2, Bm2, mb2, mm2, mW2, mw2[0], mT2)
    print(mW1, mW2, _mT1, _mT2)
    #print("->", Z1y - Z1x, Z2y - Z2x)
    mW1 = mw1[0]
    mW2 = mw2[0]
    mT1 = _mT1
    mT2 = _mT2
    ax.plot(fx1[:,1], fx1[:,0], color=col[i], linestyle = "dashed", alpha=0.8)
    ax.plot(fx2[:,1], fx2[:,0], color=col[i], linestyle = "solid", alpha=0.8)

#ax.scatter(fx1[:,1], fx1[:, 2], fx1[:,0], color='y', alpha=0.6)
#ax.set_xlim(-400, 400) 
#ax.set_ylim(-400, 400) 
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
plt.show()
