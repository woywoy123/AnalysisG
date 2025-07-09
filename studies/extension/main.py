from AnalysisG import Analysis
from AnalysisG.events.bsm_4tops import BSM4Tops
from AnalysisG.selections.neutrino.validation.validation import Validation
from nusol import *
from multiprocessing import Process
import pickle

def get_pairs(ev, name, ix): 
    px = ev.Particles[name]
    if not len(px[0]) or not len(px[1]): return None, None
    return [k for k in px[ix] if k.is_lep][0], [k for k in px[ix] if k.is_b][0]

def get_masses(nu, b, lep): 
    return ((nu + b + lep).Mass, (nu + lep).Mass) if nu is not None else (-1, -1)

def vector(inpt): return [i.vec for i in inpt]

def MPNuNu(k, metz, mx):
    #k = pickle.loads(k)
    bs, ls, ev, mW1, mT1, mW2, mT2 = k
    return DoubleNu(bs, ls, ev, mW1, mT1, mW2, mT2, metz, "-".join([str(l) for l in mx]))

ev = BSM4Tops()
sl = Validation()
sl.InterpretROOT("./ProjectName/Selections/" + sl.__name__() + "-" + ev.__name__() + "/tttt/user.tnommens.40945849._000001.output.root", "nominal")
lx = len(sl.Events)
for i in range(6,lx):
    name = "jetleptons"
    o = i
    i = sl.Events[i]
    try: nu1, nu2 = i.TruthNeutrinos
    except ValueError: continue

    l1, b1 = get_pairs(i, name, 0)
    l2, b2 = get_pairs(i, name, 1)
    if l1 is None: continue

    mt1, mw1 = get_masses(nu1, b1, l1)
    mt2, mw2 = get_masses(nu2, b2, l2)
    mtt1 = mt1



    bsx, lsx = vector((b1, b2)), vector((l1, l2))
    def skp(z, sp, m, t = 50): return m - (t-z)*sp 

    print(mw1*0.001, mt1*0.001, mw2*0.001, mt2*0.001)
    step = 0.1*1000
    it = 10000
#    mw1 = 82.6*1000
#    mw2 = 82.6*1000
#    mt1 = 172.16 * 1000
#    mt2 = 172.16 * 1000
    for l in range(0, it):
        print(nu1, nu2)
        print(nu1.px*0.001, nu1.py*0.001, nu1.pz*0.001)
        print(nu2.px*0.001, nu2.py*0.001, nu2.pz*0.001)
        exit()
        print("-----")
        dnu = DoubleNu(bsx, lsx, i.vec, mw1, mt1, mw2, skp(l, step, mt2), 0, str(l)) 
        
        xt = dnu.nunu_s()
        ns1, ns2 = xt[0], xt[1]
        for k in range(len(ns1)):
            mxt1, mws1 = get_masses(ns1[k], b1, l1)
            mxt2, mws2 = get_masses(ns2[k], b2, l2)
            print("->", abs(mxt1 - mtt1), mt2, mws1, mw2)

    exit()

    nx = MPNuNu((bsx, lsx, i.vec, mw1, mt1, mw2, mt2), 0, [o])
    xt = nx.nunu_s()
    ns1, ns2 = xt[0], xt[1]
    for k in range(len(ns1)):
        mxt1, mw1 = get_masses(ns1[k], b1, l1)
        mxt2, mw2 = get_masses(ns2[k], b2, l2)
        print("->", abs(mxt1 - mt1), mt2, mw1, mw2)

    exit()

    g = 0
    x, y, z, u = [], [], [], []
    for j in range(1):
        for t in range(1):
            for k in range(1):
                for l in range(it):
                    nx = MPNuNu((bsx, lsx, i.vec, skp(k, step*0, mw2), skp(k, step*0, mt2), skp(l, step, mw2), skp(l, step*0, mt2)), skp(l, step*0, 0), [o, j, t, k, l])

                    if nx.failed: continue
#                    print(j, t, k, l)
       #             print(j, t, k, l, "MW: ", skp(k, step, mw1), "MT: ", skp(k, step, mt1), "MW: ", skp(l, step, mw2), "MT: ", skp(l, step, mt2))
                    #print("met -> phi", nx.met_con[0], "gamma", nx.met_con[1], "nu -> phi", nx.nu_con[0], "gamma", nx.nu_con[1])
                    x.append(nx.met_con[0])
                    y.append(nx.met_con[1])
                    z.append(nx.nu_con[0])
                    u.append(nx.nu_con[1])
                    g+=1

                    xt = nx.nunu_s()
                    ns1, ns2 = xt[0], xt[1]
#                    for h in range(len(ns1)):
#                        mxt1, mxw1 = get_masses(ns1[h], b1, l1)
#                        mxt2, mxw2 = get_masses(ns2[h], b2, l2)
#                        print("->", abs(mxt1 - mt1), abs(mt2-mxt2), abs(mw1-mxw1), abs(mw2-mxw2))
#                        break
                continue
                if g < 1000: continue
                import matplotlib.pyplot as plt

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(x, z, color = "b", alpha = 0.6)
                ax.scatter(y, u, color = "g", alpha = 0.6)
                ax.scatter(y, z, color = "r", alpha = 0.6)

                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_xlim(-180, 180)
                ax.set_ylim(-180, 180)
                plt.savefig("data")
                plt.close()
                g = 0

