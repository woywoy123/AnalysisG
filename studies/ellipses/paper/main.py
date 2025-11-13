from classes import * 
from nusol import *
from visualize import *

ix = 0
idx = 1

loss  = {}
truth = {}
junk  = {}
njunk = {}
for i in DataLoader():
    if i.idx == 0: continue
    print("EVENT: ", i.idx)
    leptons  = []
    bjets    = []
    neutrino = []
    for k in i.truth_pairs:
        nu   = None
        bjet = None
        lep  = None
        for t in i.truth_pairs[k]:
            if t.hash is None: nu = t; continue
            if t.mass < 60: lep = t
            else: bjet = t
        if lep is not None:   leptons.append(lep)
        if bjets is not None: bjets.append(bjet)
        if nu is not None:    neutrino.append(nu)
    if len(neutrino) < 2 or len(leptons) < 2: continue
    break


f = figure()
f.title = "Two Ellipses Intersecting"
f.ax.grid(True)
f.ax.minorticks_on()
f.axis_label("x", r"$\vec{p}_{x}$ (GeV)")
f.axis_label("y", r"$\vec{p}_{y}$ (GeV)")
f.axis_label("z", r"$\vec{p}_{z}$ (GeV)")

#f.max_x = 100
#f.max_y = 100
#f.max_z = 100
#f.min_x = -100
#f.min_y = -100
#f.min_z = -100
f.auto_lim = True

sol1 = NuSol(bjets[0], leptons[0], f, None, (leptons[0] + neutrino[0]).mass2, (bjets[0] + leptons[0] + neutrino[0]).mass2)
sol1.color = "k-"

sol2 = NuSol(bjets[1], leptons[1], f, None, (leptons[1] + neutrino[1]).mass2, (bjets[1] + leptons[1] + neutrino[1]).mass2)
sol2.color = "b-"

sol3 = NuSol(bjets[0], leptons[0], f, None, (leptons[0] + neutrino[0]).mass2, (bjets[0] + leptons[0] + neutrino[0]).mass2)
sol3.color = "r-"

hx1 = sol1.H
hx2 = sol2.H



hinv = np.linalg.inv(hx1 / np.linalg.norm(hx1))
hinv = hinv.T.dot(hx2 / np.linalg.norm(hx2)).dot(hinv)

#sol1.plx = hinv
sol2.plx = hx1 / np.linalg.norm(hx1)
sol3.plx = hx2 / np.linalg.norm(hx2)

k = 0
es = [hx1]
from scipy.optimize import leastsq
def nus(ts): return tuple(e.dot([math.cos(t), math.sin(t), 1]) for e, t in zip(es, ts))
def residuals(params): return sum(nus(params), -np.array([neutrino[k].px, neutrino[k].py, neutrino[k].pz]))
sol2.polx.ptx, _ = leastsq(residuals, [0], ftol=5e-5, epsfcn=0.01)

k = 1
es = [hx2]
from scipy.optimize import leastsq
def nus(ts): return tuple(e.dot([math.cos(t), math.sin(t), 1]) for e, t in zip(es, ts))
def residuals(params): return sum(nus(params), -np.array([neutrino[k].px, neutrino[k].py, neutrino[k].pz]))
sol3.polx.ptx, _ = leastsq(residuals, [0], ftol=5e-5, epsfcn=0.01)

f.add_object("ellipse-1", sol1)
f.add_object("ellipse-2", sol2)
f.add_object("ellipse-3", sol3)

f.show()




