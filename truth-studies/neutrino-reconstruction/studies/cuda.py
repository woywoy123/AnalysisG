from AnalysisG.IO import PickleObject, UnpickleObject
from AnalysisG.Templates import SelectionTemplate
from pyc.interface import pyc_cuda, pyc_tensor
from .nusol import doubleNeutrinoSolutions
import torch
import time
import vector

class Performance(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)

        self.truth_container = {
                "event"    : {"metx" : [], "mety" : [], "nu1" : [], "nu2" : []},
                "children" : {"b1"   : [], "b2"   : [], "l1" : [], "l2" : [], "mt1" : [], "mw1" : [], "mt2" : [], "mw2" : []},
                "truthjet" : {"b1"   : [], "b2"   : [], "l1" : [], "l2" : [], "mt1" : [], "mw1" : [], "mt2" : [], "mw2" : []},
                "jets"     : {"b1"   : [], "b2"   : [], "l1" : [], "l2" : [], "mt1" : [], "mw1" : [], "mt2" : [], "mw2" : []},
                "detector" : {"b1"   : [], "b2"   : [], "l1" : [], "l2" : [], "mt1" : [], "mw1" : [], "mt2" : [], "mw2" : []}
        }

        self.combinatorial = {
                "event"    : [],
                "children" : {"edge_index" : [], "batch" : [], "pmc" : [], "pid" : []},
                "truthjet" : {"edge_index" : [], "batch" : [], "pmc" : [], "pid" : []},
                "jets"     : {"edge_index" : [], "batch" : [], "pmc" : [], "pid" : []},
                "detector" : {"edge_index" : [], "batch" : [], "pmc" : [], "pid" : []}
        }

    def Selection(self, event):
        res = 0
        for t in event.Tops:
            x = [1 for c in t.Children if c.is_lep]
            if not len(x): continue
            res += 1
        if res == 2: return True
        return False

    def as_vec(self, inpt):
        if inpt is None: return False
        dct = {"pt" : inpt.pt, "eta" : inpt.eta, "phi" : inpt.phi, "energy" : inpt.e}
        return vector.obj(**dct)

    def make_original(self, bs, ls, met_x, met_y, mT, mW):
        try: nu_ref = doubleNeutrinoSolutions(bs, ls, met_x, met_y, mW, mT)
        except: return False
        nu_ref = nu_ref.nunu_s
        if not len(nu_ref): return False
        for sols in nu_ref: return (self.MakeNu(sols[0]), self.MakeNu(sols[1]))


    def make_combination(self, container, key, dic):
        l = len(container)
        edge1 = []
        edge2 = []
        for i in range(l):
            for j in range(l):
                edge1 += [i]
                edge2 += [j]
        ed1 = torch.tensor(edge1)
        ed2 = torch.tensor(edge2)
        dic[key]["edge_index"] += [torch.cat([ed1.view(1, -1), ed2.view(1, -1)], 0)]
        dic[key]["batch"] += [torch.tensor([0 for i in range(l)])]

        pmc = []
        pid = []
        for k in container:
            pmc += [[k.px, k.py, k.pz, k.e]]
            pid += [[k.is_lep, k.is_b]]
        dic[key]["pmc"] += [torch.tensor(pmc, dtype = torch.float64)]
        dic[key]["pid"] += [torch.tensor(pid)]

    def Strategy(self, event):
        lep_tops = []
        for t in event.Tops:
            x = [1 for c in t.Children if c.is_lep]
            if not len(x): continue
            lep_tops.append(t)

        t1, t2 = lep_tops
        b1  = [i for i in t1.Children if i.is_b][0]
        l1  = [i for i in t1.Children if i.is_lep and not i.is_nu][0]
        nu1 = [i for i in t1.Children if i.is_nu][0]

        b2  = [i for i in t2.Children if i.is_b][0]
        l2  = [i for i in t2.Children if i.is_lep and not i.is_nu][0]
        nu2 = [i for i in t2.Children if i.is_nu][0]

        met = event.met
        phi = event.met_phi

        met_x = self.Px(met, phi)
        met_y = self.Py(met, phi)

        tjb1 = None
        for j in t1.TruthJets:
            # assert that a truth jet has no more than 2 top contributions
            if len(j.Tops) > 1: continue
            for p in j.Parton:
                if b1 not in p.Parent: continue
                tjb1 = j
            if tjb1 is None: continue
            break

        tjb2 = None
        for j in t2.TruthJets:
            # assert that a truth jet has no more than 2 top contributions
            if len(j.Tops) > 1: continue
            for p in j.Parton:
                if b2 not in p.Parent: continue
                tjb2 = j
            if tjb2 is None: continue
            break

        jb1 = None
        for j in t1.Jets:
            # assert that a truth jet has no more than 2 top contributions
            if len(j.Tops) > 1: continue
            for p in j.Parton:
                if b1 not in p.Parent: continue
                jb1 = j
            if jb1 is None: continue
            break

        jb2 = None
        for j in t2.Jets:
            # assert that a truth jet has no more than 2 top contributions
            if len(j.Tops) > 1: continue
            for p in j.Parton:
                if b2 not in p.Parent: continue
                jb2 = j
            if jb2 is None: continue
            break

        lep1, lep2 = l1.Children, l2.Children
        if len(lep1) and len(lep2): lep1, lep2 = lep1[0], lep2[0]
        else: lep1, lep2 = None, None


        self.truth_container["event"]["metx"] += [met_x]
        self.truth_container["event"]["mety"] += [met_y]

        self.truth_container["event"]["nu1"] += [nu1]
        self.truth_container["event"]["nu2"] += [nu2]

        self.truth_container["children"]["b1"] += [b1]
        self.truth_container["children"]["b2"] += [b2]

        self.truth_container["children"]["l1"] += [l1]
        self.truth_container["children"]["l2"] += [l2]

        self.truth_container["children"]["mt1"] += [(b1 + l1 + nu1).Mass]
        self.truth_container["children"]["mt2"] += [(b2 + l2 + nu2).Mass]
        self.truth_container["children"]["mw1"] += [(l1 + nu1).Mass]
        self.truth_container["children"]["mw2"] += [(l2 + nu2).Mass]

        self.truth_container["truthjet"]["b1"] += [tjb1]
        self.truth_container["truthjet"]["b2"] += [tjb2]
        self.truth_container["truthjet"]["l1"] += [l1]
        self.truth_container["truthjet"]["l2"] += [l2]
        if tjb1 is not None and tjb2 is not None:
            self.truth_container["truthjet"]["mt1"] += [(tjb1 + l1 + nu1).Mass]
            self.truth_container["truthjet"]["mt2"] += [(tjb2 + l2 + nu2).Mass]
            self.truth_container["truthjet"]["mw1"] += [(l1 + nu1).Mass]
            self.truth_container["truthjet"]["mw2"] += [(l2 + nu2).Mass]
        else:
            self.truth_container["truthjet"]["mt1"] += [None]
            self.truth_container["truthjet"]["mt2"] += [None]
            self.truth_container["truthjet"]["mw1"] += [None]
            self.truth_container["truthjet"]["mw2"] += [None]

        self.truth_container["jets"]["b1"] += [jb1]
        self.truth_container["jets"]["b2"] += [jb2]
        self.truth_container["jets"]["l1"] += [l1]
        self.truth_container["jets"]["l2"] += [l2]
        if jb1 is not None and jb2 is not None:
            self.truth_container["jets"]["mt1"] += [(jb1 + l1 + nu1).Mass]
            self.truth_container["jets"]["mt2"] += [(jb2 + l2 + nu2).Mass]
            self.truth_container["jets"]["mw1"] += [(l1 + nu1).Mass]
            self.truth_container["jets"]["mw2"] += [(l2 + nu2).Mass]
        else:
            self.truth_container["jets"]["mt1"] += [None]
            self.truth_container["jets"]["mt2"] += [None]
            self.truth_container["jets"]["mw1"] += [None]
            self.truth_container["jets"]["mw2"] += [None]

        self.truth_container["detector"]["b1"] += [jb1]
        self.truth_container["detector"]["b2"] += [jb2]
        self.truth_container["detector"]["l1"] += [lep1]
        self.truth_container["detector"]["l2"] += [lep2]
        if jb1 is not None and jb2 is not None and lep1 is not None and lep2 is not None:
            self.truth_container["detector"]["mt1"] += [(jb1 + lep1 + nu1).Mass]
            self.truth_container["detector"]["mt2"] += [(jb2 + lep2 + nu2).Mass]
            self.truth_container["detector"]["mw1"] += [(lep1 + nu1).Mass]
            self.truth_container["detector"]["mw2"] += [(lep2 + nu2).Mass]
        else:
            self.truth_container["detector"]["mt1"] += [None]
            self.truth_container["detector"]["mt2"] += [None]
            self.truth_container["detector"]["mw1"] += [None]
            self.truth_container["detector"]["mw2"] += [None]


        children_ = [k for k in event.TopChildren if not k.is_nu]
        truthjet_ = [k for k in event.TruthJets] + [k for k in children_ if k.is_lep]
        jet_ = [k for k in event.Jets] + [k for k in children_ if k.is_lep]
        recos_ = event.DetectorObjects

        self.combinatorial["event"] += [torch.tensor([met_x, met_y], dtype = torch.float64).view(-1, 2)]
        self.make_combination(children_, "children", self.combinatorial)
        self.make_combination(truthjet_, "truthjet", self.combinatorial)
        self.make_combination(jet_     , "jets"    , self.combinatorial)
        self.make_combination(recos_   , "detector", self.combinatorial)

def measure_pyc(packet, key, perfx_, res, use_cuda):
    truth  = packet["truth_data"]
    aten   = packet["aten"][key]
    bools  = torch.tensor(aten["bools"], dtype = torch.bool)

    mass_W_C1 = aten["pmc"][2] + truth["aten_nu1"][bools]
    mass_W_C2 = aten["pmc"][3] + truth["aten_nu2"][bools]
    mass_T_C1 = aten["pmc"][2] + truth["aten_nu1"][bools] + aten["pmc"][0]
    mass_T_C2 = aten["pmc"][3] + truth["aten_nu2"][bools] + aten["pmc"][1]

    mass_w_c  = pyc_tensor.combined.physics.cartesian.M(mass_W_C1)
    mass_w_c += pyc_tensor.combined.physics.cartesian.M(mass_W_C2)

    mass_t_c  = pyc_tensor.combined.physics.cartesian.M(mass_T_C1)
    mass_t_c += pyc_tensor.combined.physics.cartesian.M(mass_T_C2)
    masses = torch.cat([mass_w_c/2, mass_t_c/2, torch.zeros_like(mass_t_c)], -1)

    pmc_b1, pmc_b2 = aten["pmc"][0], aten["pmc"][1]
    pmc_l1, pmc_l2 = aten["pmc"][2], aten["pmc"][3]
    met_xy = torch.cat(perfx_.combinatorial["event"], 0)[bools]

    if use_cuda:
        nusol_pyc = pyc_cuda.nusol.NuNu
        pmc_b1, pmc_b2 = pmc_b1.to(device = "cuda:0"), pmc_b2.to(device = "cuda:0")
        pmc_l1, pmc_l2 = pmc_l1.to(device = "cuda:0"), pmc_l2.to(device = "cuda:0")
        met_xy, masses = met_xy.to(device = "cuda:0"), masses.to(device = "cuda:0")
        nu1_ = truth["aten_nu1"].to(device = "cuda:0")
        nu2_ = truth["aten_nu2"].to(device = "cuda:0")
        bools = bools.to(device = "cuda:0")
    else:
        nusol_pyc = pyc_tensor.nusol.NuNu
        nu1_, nu2_ = truth["aten_nu1"], truth["aten_nu2"]

    t1 = time.time()
    nu1_sol, nu2_sol, diag, _, _, _, sol = nusol_pyc(pmc_b1, pmc_b2, pmc_l1, pmc_l2, met_xy, masses, 1e-10)
    t2 = time.time() - t1

    msk = diag > 0
    msk_ = msk.sum(-1) > 0
    best_sol = msk.cumsum(-1) == 1

    diags = diag[best_sol]
    nu1_sol = nu1_sol[best_sol]
    nu2_sol = nu2_sol[best_sol]

    nu1_sol = torch.cat([nu1_sol, torch.sum(nu1_sol.pow(2), -1, keepdim = True).pow(0.5)], -1)
    nu2_sol = torch.cat([nu2_sol, torch.sum(nu2_sol.pow(2), -1, keepdim = True).pow(0.5)], -1)

    nu1_delta = (nu1_sol - nu1_[bools][msk_])/1000
    nu2_delta = (nu2_sol - nu2_[bools][msk_])/1000

    nusol_mt1 = pyc_tensor.combined.physics.cartesian.M(nu1_sol + pmc_b1[msk_] + pmc_l1[msk_])/1000
    nusol_mt2 = pyc_tensor.combined.physics.cartesian.M(nu2_sol + pmc_b2[msk_] + pmc_l2[msk_])/1000

    res[key]["mt"] = torch.cat([nusol_mt1, nusol_mt2], 0).view(-1).tolist()
    res[key]["px"] = torch.cat([nu1_delta[:, 0], nu2_delta[:, 0]], -1).view(-1).tolist()
    res[key]["py"] = torch.cat([nu1_delta[:, 1], nu2_delta[:, 1]], -1).view(-1).tolist()
    res[key]["pz"] = torch.cat([nu1_delta[:, 2], nu2_delta[:, 2]], -1).view(-1).tolist()
    res[key]["e"]  = torch.cat([nu1_delta[:, 3], nu2_delta[:, 3]], -1).view(-1).tolist()
    res[key]["time"] = float(t2/met_xy.size(0))
    return res

def measure_cuda_combinatorial(key, perfx_, result, algo):
    truth = perfx_.truth_container
    lex = len(perfx_.combinatorial["event"])
    for i in range(lex):
        if not (i%100): print(round(float(i/lex), 3)*100, key)
        nu_t1, nu_t2 = truth["event"]["nu1"][i], truth["event"]["nu2"][i]
        if truth[key]["mt1"][i] is None: continue
        b1, b2 = truth[key]["b1"][i], truth[key]["b2"][i]
        l1, l2 = truth[key]["l1"][i], truth[key]["l2"][i]

        met_xy = perfx_.combinatorial["event"][i].to(device = "cuda:0")
        pmc = perfx_.combinatorial[key]["pmc"][i].to(device = "cuda:0")
        pid = perfx_.combinatorial[key]["pid"][i].to(device = "cuda:0")
        batch = perfx_.combinatorial[key]["batch"][i].to(device = "cuda:0")
        edge_index = perfx_.combinatorial[key]["edge_index"][i].to(device = "cuda:0")

        res = algo(edge_index, batch, pmc, pid, met_xy)
        if res["combination"].sum(-1) == 0: continue

        nu1_f, nu2_f = res["nu_1f"], res["nu_2f"]
        mass1, mass2 = res["masses_nu1"], res["masses_nu2"]
        mass1, mass2 = mass1[0].tolist(), mass2[0].tolist()
        m_w1, m_t1, m_w2, m_t2 = mass1[0], mass1[1], mass2[0], mass2[1]
        result[key]["mt"] += [m_t1/1000, m_t2/1000]
        result[key]["mw"] += [m_w1/1000, m_w2/1000]
        result[key]["tru_mt"] += [(nu_t1 + b1 + l1).Mass/1000, (nu_t2 + b2 + l2).Mass/1000]
        result[key]["tru_wt"] += [(nu_t1 + l1).Mass/1000, (nu_t2 + l2).Mass/1000]

        nu1_, nu2_ = perfx_.MakeNu(nu1_f[0]), perfx_.MakeNu(nu2_f[0])
        result[key]["mt_sol"] += [(nu1_ + b1 + l1).Mass/1000,(nu2_ + b2 + l2).Mass/1000]
        result[key]["px"] += [(nu1_.px - nu_t1.px)/1000, (nu2_.px - nu_t2.px)/1000]
        result[key]["py"] += [(nu1_.py - nu_t1.py)/1000, (nu2_.py - nu_t2.py)/1000]
        result[key]["pz"] += [(nu1_.pz - nu_t1.pz)/1000, (nu2_.pz - nu_t2.pz)/1000]
        result[key]["e"]  += [(nu1_.e  - nu_t1.e )/1000, (nu2_.e  - nu_t2.e )/1000]
    return result





def get_samples(ana):
    lex = 1
    ley = 1
    prf = Performance()
    t = 0
    x = -1
    if ley > 0:
        for event in ana:
            if x == -1: x = len(ana)
            t += 1
            if not prf.Selection(event): continue
            prf.Strategy(event)
            if t%100: continue
            print(round(float(t/x)*100, 3))

        PickleObject(prf, "tmp.pkl")

    # reference compilation
    truth_data = {"mtc" : [], "mt_tj" : [], "mt_j" : [], "mt_rlj" : [], "aten_nu1" : None, "aten_nu2" : None}

    reference = {}
    reference |= {"children" : {"time" : [], "mt" : [], "px" : [], "py" : [], "pz" : [], "e" : []}}
    reference |= {"truthjet" : {"time" : [], "mt" : [], "px" : [], "py" : [], "pz" : [], "e" : []}}
    reference |= {"jets"     : {"time" : [], "mt" : [], "px" : [], "py" : [], "pz" : [], "e" : []}}
    reference |= {"detector" : {"time" : [], "mt" : [], "px" : [], "py" : [], "pz" : [], "e" : []}}

    aten = {}
    aten |= {"children" : {"time" : [], "mt" : [], "pmc" : None, "bools" : []}}
    aten |= {"truthjet" : {"time" : [], "mt" : [], "pmc" : None, "bools" : []}}
    aten |= {"jets"     : {"time" : [], "mt" : [], "pmc" : None, "bools" : []}}
    aten |= {"detector" : {"time" : [], "mt" : [], "pmc" : None, "bools" : []}}


    if lex > 0:
        prf = UnpickleObject("tmp.pkl")
        truth = prf.truth_container
        lex = len(truth["event"]["metx"])

    for k in range(lex):
        print(k/lex)
        metx, mety = truth["event"]["metx"][k], truth["event"]["mety"][k]
        nu_t1, nu_t2 = truth["event"]["nu1"][k], truth["event"]["nu2"][k]

        nu_ten1 = torch.tensor([[nu_t1.px, nu_t1.py, nu_t1.pz, nu_t1.e]], dtype = torch.float64)
        nu_ten2 = torch.tensor([[nu_t2.px, nu_t2.py, nu_t2.pz, nu_t2.e]], dtype = torch.float64)
        if truth_data["aten_nu1"] is None:
            truth_data["aten_nu1"] = nu_ten1
            truth_data["aten_nu2"] = nu_ten2
        else:
            truth_data["aten_nu1"] = torch.cat([truth_data["aten_nu1"], nu_ten1], 0)
            truth_data["aten_nu2"] = torch.cat([truth_data["aten_nu2"], nu_ten2], 0)

        # children
        b1, b2 = truth["children"]["b1"][k], truth["children"]["b2"][k]
        l1, l2 = truth["children"]["l1"][k], truth["children"]["l2"][k]
        mt = (truth["children"]["mt1"][k] + truth["children"]["mt2"][k])/2
        mw = (truth["children"]["mw1"][k] + truth["children"]["mw2"][k])/2
        bs = (prf.as_vec(b1), prf.as_vec(b2))
        ls = (prf.as_vec(l1), prf.as_vec(l2))

        t1 = time.time()
        nus = prf.make_original(bs, ls, metx, mety, mt, mw)
        t2 = time.time() - t1
        reference["children"]["time"] += [t2]
        if nus is not False:
            reference["children"]["mt"] += [(nus[0] + b1 + l1).Mass/1000, (nus[1] + b2 + l2).Mass/1000]
            reference["children"]["px"] += [(nus[0].px - nu_t1.px)/1000, (nus[1].px - nu_t2.px)/1000]
            reference["children"]["py"] += [(nus[0].py - nu_t1.py)/1000, (nus[1].py - nu_t2.py)/1000]
            reference["children"]["pz"] += [(nus[0].pz - nu_t1.pz)/1000, (nus[1].pz - nu_t2.pz)/1000]
            reference["children"]["e"]  += [(nus[0].e  - nu_t1.e )/1000, (nus[1].e  - nu_t2.e )/1000]

        truth_data["mtc"] += [(b1 + l1 + nu_t1).Mass/1000, (b2 + l2 + nu_t2).Mass/1000]

        b1t = torch.tensor([[b1.px, b1.py, b1.pz, b1.e]], dtype = torch.float64)
        b2t = torch.tensor([[b2.px, b2.py, b2.pz, b2.e]], dtype = torch.float64)
        l1t = torch.tensor([[l1.px, l1.py, l1.pz, l1.e]], dtype = torch.float64)
        l2t = torch.tensor([[l2.px, l2.py, l2.pz, l2.e]], dtype = torch.float64)
        aten["children"]["bools"] += [True]

        if aten["children"]["pmc"] is None: aten["children"]["pmc"] = [b1t, b2t, l1t, l2t]
        else:
            aten["children"]["pmc"][0] = torch.cat([aten["children"]["pmc"][0], b1t], 0)
            aten["children"]["pmc"][1] = torch.cat([aten["children"]["pmc"][1], b2t], 0)
            aten["children"]["pmc"][2] = torch.cat([aten["children"]["pmc"][2], l1t], 0)
            aten["children"]["pmc"][3] = torch.cat([aten["children"]["pmc"][3], l2t], 0)

        # truthjet
        b1, b2 = truth["truthjet"]["b1"][k], truth["truthjet"]["b2"][k]
        l1, l2 = truth["truthjet"]["l1"][k], truth["truthjet"]["l2"][k]
        if truth["truthjet"]["mt1"][k] is not None:
            aten["truthjet"]["bools"] += [True]

            mt = (truth["truthjet"]["mt1"][k] + truth["truthjet"]["mt2"][k])/2
            mw = (truth["truthjet"]["mw1"][k] + truth["truthjet"]["mw2"][k])/2
            bs = (prf.as_vec(b1), prf.as_vec(b2))
            ls = (prf.as_vec(l1), prf.as_vec(l2))

            t1 = time.time()
            nus = prf.make_original(bs, ls, metx, mety, mt, mw)
            t2 = time.time() - t1
            reference["truthjet"]["time"] += [t2]
            if nus is not False:
                reference["truthjet"]["mt"] += [(nus[0] + b1 + l1).Mass/1000, (nus[1] + b2 + l2).Mass/1000]
                reference["truthjet"]["px"] += [(nus[0].px - nu_t1.px)/1000, (nus[1].px - nu_t2.px)/1000]
                reference["truthjet"]["py"] += [(nus[0].py - nu_t1.py)/1000, (nus[1].py - nu_t2.py)/1000]
                reference["truthjet"]["pz"] += [(nus[0].pz - nu_t1.pz)/1000, (nus[1].pz - nu_t2.pz)/1000]
                reference["truthjet"]["e"]  += [(nus[0].e  - nu_t1.e )/1000, (nus[1].e  - nu_t2.e )/1000]

            b1t = torch.tensor([[b1.px, b1.py, b1.pz, b1.e]], dtype = torch.float64)
            b2t = torch.tensor([[b2.px, b2.py, b2.pz, b2.e]], dtype = torch.float64)
            l1t = torch.tensor([[l1.px, l1.py, l1.pz, l1.e]], dtype = torch.float64)
            l2t = torch.tensor([[l2.px, l2.py, l2.pz, l2.e]], dtype = torch.float64)
            if aten["truthjet"]["pmc"] is None: aten["truthjet"]["pmc"] = [b1t, b2t, l1t, l2t]
            else:
                aten["truthjet"]["pmc"][0] = torch.cat([aten["truthjet"]["pmc"][0], b1t], 0)
                aten["truthjet"]["pmc"][1] = torch.cat([aten["truthjet"]["pmc"][1], b2t], 0)
                aten["truthjet"]["pmc"][2] = torch.cat([aten["truthjet"]["pmc"][2], l1t], 0)
                aten["truthjet"]["pmc"][3] = torch.cat([aten["truthjet"]["pmc"][3], l2t], 0)
            truth_data["mt_tj"]  += [(b1 + l1 + nu_t1).Mass/1000, (b2 + l2 + nu_t2).Mass/1000]
        else: aten["truthjet"]["bools"] += [False]

        # jets
        b1, b2 = truth["jets"]["b1"][k], truth["jets"]["b2"][k]
        l1, l2 = truth["jets"]["l1"][k], truth["jets"]["l2"][k]
        if truth["jets"]["mt1"][k] is not None:
            aten["jets"]["bools"] += [True]

            mt = (truth["jets"]["mt1"][k] + truth["jets"]["mt2"][k])/2
            mw = (truth["jets"]["mw1"][k] + truth["jets"]["mw2"][k])/2
            bs = (prf.as_vec(b1), prf.as_vec(b2))
            ls = (prf.as_vec(l1), prf.as_vec(l2))

            t1 = time.time()
            nus = prf.make_original(bs, ls, metx, mety, mt, mw)
            t2 = time.time() - t1
            reference["jets"]["time"] += [t2]
            if nus is not False:
                reference["jets"]["mt"] += [(nus[0] + b1 + l1).Mass/1000, (nus[1] + b2 + l2).Mass/1000]
                reference["jets"]["px"] += [(nus[0].px -  nu_t1.px)/1000, (nus[1].px -  nu_t2.px)/1000]
                reference["jets"]["py"] += [(nus[0].py -  nu_t1.py)/1000, (nus[1].py -  nu_t2.py)/1000]
                reference["jets"]["pz"] += [(nus[0].pz -  nu_t1.pz)/1000, (nus[1].pz -  nu_t2.pz)/1000]
                reference["jets"]["e"]  += [(nus[0].e  -  nu_t1.e )/1000, (nus[1].e  -  nu_t2.e )/1000]


            b1t = torch.tensor([[b1.px, b1.py, b1.pz, b1.e]], dtype = torch.float64)
            b2t = torch.tensor([[b2.px, b2.py, b2.pz, b2.e]], dtype = torch.float64)
            l1t = torch.tensor([[l1.px, l1.py, l1.pz, l1.e]], dtype = torch.float64)
            l2t = torch.tensor([[l2.px, l2.py, l2.pz, l2.e]], dtype = torch.float64)
            if aten["jets"]["pmc"] is None: aten["jets"]["pmc"] = [b1t, b2t, l1t, l2t]
            else:
                aten["jets"]["pmc"][0] = torch.cat([aten["jets"]["pmc"][0], b1t], 0)
                aten["jets"]["pmc"][1] = torch.cat([aten["jets"]["pmc"][1], b2t], 0)
                aten["jets"]["pmc"][2] = torch.cat([aten["jets"]["pmc"][2], l1t], 0)
                aten["jets"]["pmc"][3] = torch.cat([aten["jets"]["pmc"][3], l2t], 0)
            truth_data["mt_j"] += [(b1 + l1 + nu_t1).Mass/1000, (b2 + l2 + nu_t2).Mass/1000]
        else: aten["jets"]["bools"] += [False]

        # jets + leptons: detector
        b1, b2 = truth["detector"]["b1"][k], truth["detector"]["b2"][k]
        l1, l2 = truth["detector"]["l1"][k], truth["detector"]["l2"][k]
        if truth["detector"]["mt1"][k] is not None:
            aten["detector"]["bools"] += [True]

            mt = (truth["detector"]["mt1"][k] + truth["detector"]["mt2"][k])/2
            mw = (truth["detector"]["mw1"][k] + truth["detector"]["mw2"][k])/2
            bs = (prf.as_vec(b1), prf.as_vec(b2))
            ls = (prf.as_vec(l1), prf.as_vec(l2))

            t1 = time.time()
            nus = prf.make_original(bs, ls, metx, mety, mt, mw)
            t2 = time.time() - t1
            reference["detector"]["time"] += [t2]
            if nus is not False:
                reference["detector"]["mt"] += [(nus[0] + b1 + l1).Mass/1000, (nus[1] + b2 + l2).Mass/1000]
                reference["detector"]["px"] += [(nus[0].px - nu_t1.px)/1000, (nus[1].px - nu_t2.px)/1000]
                reference["detector"]["py"] += [(nus[0].py - nu_t1.py)/1000, (nus[1].py - nu_t2.py)/1000]
                reference["detector"]["pz"] += [(nus[0].pz - nu_t1.pz)/1000, (nus[1].pz - nu_t2.pz)/1000]
                reference["detector"]["e"]  += [(nus[0].e  - nu_t1.e )/1000, (nus[1].e  - nu_t2.e )/1000]

            b1t = torch.tensor([[b1.px, b1.py, b1.pz, b1.e]], dtype = torch.float64)
            b2t = torch.tensor([[b2.px, b2.py, b2.pz, b2.e]], dtype = torch.float64)
            l1t = torch.tensor([[l1.px, l1.py, l1.pz, l1.e]], dtype = torch.float64)
            l2t = torch.tensor([[l2.px, l2.py, l2.pz, l2.e]], dtype = torch.float64)
            if aten["detector"]["pmc"] is None: aten["detector"]["pmc"] = [b1t, b2t, l1t, l2t]
            else:
                aten["detector"]["pmc"][0] = torch.cat([aten["detector"]["pmc"][0], b1t], 0)
                aten["detector"]["pmc"][1] = torch.cat([aten["detector"]["pmc"][1], b2t], 0)
                aten["detector"]["pmc"][2] = torch.cat([aten["detector"]["pmc"][2], l1t], 0)
                aten["detector"]["pmc"][3] = torch.cat([aten["detector"]["pmc"][3], l2t], 0)
            truth_data["mt_rlj"] += [(b1 + l1 + nu_t1).Mass/1000, (b2 + l2 + nu_t2).Mass/1000]
        else: aten["detector"]["bools"] += [False]

    if lex > 0:
        packet = {"truth_data" : truth_data, "reference" : reference, "aten" : aten}
        PickleObject(packet, "data_res.pkl")


    perfx_ = UnpickleObject("tmp.pkl")
    packet = UnpickleObject("data_res.pkl")

    reference = packet["reference"]
    reference["children"]["time"] = sum(reference["children"]["time"])/len(reference["children"]["time"])
    reference["truthjet"]["time"] = sum(reference["truthjet"]["time"])/len(reference["truthjet"]["time"])
    reference["jets"]["time"] = sum(reference["jets"]["time"])/len(reference["jets"]["time"])
    reference["detector"]["time"] = sum(reference["detector"]["time"])/len(reference["detector"]["time"])

    pyc_nusol_ten = {}
    pyc_nusol_ten |= {"children" : {"time" : [], "mt" : [], "px" : [], "py" : [], "pz" : [], "e" : []}}
    pyc_nusol_ten |= {"truthjet" : {"time" : [], "mt" : [], "px" : [], "py" : [], "pz" : [], "e" : []}}
    pyc_nusol_ten |= {"jets"     : {"time" : [], "mt" : [], "px" : [], "py" : [], "pz" : [], "e" : []}}
    pyc_nusol_ten |= {"detector" : {"time" : [], "mt" : [], "px" : [], "py" : [], "pz" : [], "e" : []}}

    pyc_nusol_ten = measure_pyc(packet, "children", perfx_, pyc_nusol_ten, False)
    pyc_nusol_ten = measure_pyc(packet, "truthjet", perfx_, pyc_nusol_ten, False)
    pyc_nusol_ten = measure_pyc(packet, "jets"    , perfx_, pyc_nusol_ten, False)
    pyc_nusol_ten = measure_pyc(packet, "detector", perfx_, pyc_nusol_ten, False)

    pyc_nusol_cu = {}
    pyc_nusol_cu |= {"children" : {"time" : [], "mt" : [], "px" : [], "py" : [], "pz" : [], "e" : []}}
    pyc_nusol_cu |= {"truthjet" : {"time" : [], "mt" : [], "px" : [], "py" : [], "pz" : [], "e" : []}}
    pyc_nusol_cu |= {"jets"     : {"time" : [], "mt" : [], "px" : [], "py" : [], "pz" : [], "e" : []}}
    pyc_nusol_cu |= {"detector" : {"time" : [], "mt" : [], "px" : [], "py" : [], "pz" : [], "e" : []}}

    pyc_nusol_cu = measure_pyc(packet, "children", perfx_, pyc_nusol_cu, True)
    pyc_nusol_cu = measure_pyc(packet, "truthjet", perfx_, pyc_nusol_cu, True)
    pyc_nusol_cu = measure_pyc(packet, "jets"    , perfx_, pyc_nusol_cu, True)
    pyc_nusol_cu = measure_pyc(packet, "detector", perfx_, pyc_nusol_cu, True)


    pyc_nusol_comb = {}
    pyc_nusol_comb |= {"children" : {"mt_sol" : [], "mt" : [], "mw" : [], "px" : [], "py" : [], "pz" : [], "e" : [], "tru_mt" : [], "tru_wt": []}}
    pyc_nusol_comb |= {"truthjet" : {"mt_sol" : [], "mt" : [], "mw" : [], "px" : [], "py" : [], "pz" : [], "e" : [], "tru_mt" : [], "tru_wt": []}}
    pyc_nusol_comb |= {"jets"     : {"mt_sol" : [], "mt" : [], "mw" : [], "px" : [], "py" : [], "pz" : [], "e" : [], "tru_mt" : [], "tru_wt": []}}
    pyc_nusol_comb |= {"detector" : {"mt_sol" : [], "mt" : [], "mw" : [], "px" : [], "py" : [], "pz" : [], "e" : [], "tru_mt" : [], "tru_wt": []}}

    truth_container = perfx_.truth_container
    algo = torch.jit.script(pyc_cuda.nusol.combinatorial)
    pyc_nusol_comb = measure_cuda_combinatorial("children", perfx_, pyc_nusol_comb, algo)
    pyc_nusol_comb = measure_cuda_combinatorial("truthjet", perfx_, pyc_nusol_comb, algo)
    pyc_nusol_comb = measure_cuda_combinatorial("jets"    , perfx_, pyc_nusol_comb, algo)
    pyc_nusol_comb = measure_cuda_combinatorial("detector", perfx_, pyc_nusol_comb, algo)

    data = {
            "reference" : reference, "pyc_tensor" : pyc_nusol_ten,
            "pyc_cuda" : pyc_nusol_cu, "pyc_combinatorial" : pyc_nusol_comb
    }
    PickleObject(data, "results.pkl")
