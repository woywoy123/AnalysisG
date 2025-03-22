from AnalysisG.core.plotting import TH1F, TH2F
from helper import *

def enforce_sym(p, m, pgd):
    if "gamma" in p: return False
    return True

def enforce_sym_tj(p, m, pgd):
    absx = set([abs(i) for i in pgd])
    if p.count("o") >= 2: return False
    if p.count("3"): return False
    if len(absx) < 3: return False
    return True


def Tabular(topx, target, title, mode, rec = None):

    print("---------------------- " + title + " ------------------------ ")
    if target is None: 
        print("Top-Multiplicity", mrg(topx["counts"]))
        print("Initial Number of Tops", mrg(topx["ntops"]))
    else: 
        if "counts" in target: print("Top-Multiplicity", target["counts"])
        print(mode + " Number of Tops", mrg(target["ntops"]), " | Loss (%)", loss(topx["ntops"], target["ntops"], 4))

    if target is None:
        print("Leptonic", mrg(topx["leptonic"]), " | (%)",  pc(topx["ntops"], topx["leptonic"], 4))
        print("Hadronic", mrg(topx["hadronic"]), " | (%)",  pc(topx["ntops"], topx["hadronic"], 4))
        return 
    print("Leptonic                ", mrg(target["leptonic"]), " | Loss (%)", loss(topx["leptonic"], target["leptonic"], 4))
    print("Hadronic                ", mrg(target["hadronic"]), " | Loss (%)", loss(topx["hadronic"], target["hadronic"], 4))

def TabularFlow(topx, stat, title, mode, min_, max_):
    print("---------------------- " + title + " ------------------------ ")
    a_tops,  o_tops,  u_tops = (sum(stat["tops-merged"][k]["all"].values()) for k in ["domain", "over", "under"])
    l_tops, ol_tops, ul_tops = (sum(stat["tops-merged"][k]["leptonic"].values()) for k in ["domain", "over", "under"])
    h_tops, oh_tops, uh_tops = (sum(stat["tops-merged"][k]["hadronic"].values()) for k in ["domain", "over", "under"])

    at_tops = a_tops + o_tops + u_tops
    al_tops = l_tops + ol_tops + ul_tops
    ah_tops = h_tops + oh_tops + uh_tops

    print("____________________ Raw Values ____________________")
    print(mode + " Number of Tops (" + str(min_) + " -> " + str(max_) + "):", mrg(a_tops), " | Overflow: ", mrg(o_tops), " Underflow: ", mrg(u_tops))
    print("Leptonically Matched:               ", mrg(l_tops), " | Overflow: ", mrg(ol_tops), " Underflow: ", mrg(ul_tops))
    print("Hadronically Matched:               ", mrg(h_tops), " | Overflow: ", mrg(oh_tops), " Underflow: ", mrg(uh_tops))

    print("____________________ Percentage ____________________")
    print("All:                                 ", pc(at_tops, a_tops), " | Overflow: ", pc(at_tops, o_tops) , " Underflow: ", pc(at_tops, u_tops))
    print("Leptonically Matched (lep/all_leps): ", pc(al_tops, l_tops), " | Overflow: ", pc(al_tops, ol_tops), " Underflow: ", pc(al_tops, ul_tops))
    print("Hadronically Matched (had/all_hads): ", pc(ah_tops, h_tops), " | Overflow: ", pc(ah_tops, oh_tops), " Underflow: ", pc(ah_tops, uh_tops))

    print("____________________ Loss ____________________")
    print("All: ", loss(topx["ntops"], a_tops, 4), " | Leptonically Matched: ", loss(topx["leptonic"], l_tops, 4), " | Hadronically Matched: ", loss(topx["hadronic"], h_tops, 4))



def entry_point(fancy, mode, pth, tree, plt = True):



    data     = fetch_all(pth, tree, ["top-partons", "top-children", "top-truthjet"])
    tops     = fetch_data(data, "top-partons")
    children = fetch_data(data, "top-children")
    truthjet = fetch_data(data, "top-truthjet")

    topx  = top_decay_stats(tops    ,  "p_ntops",  "p_ltops",  "p_htops",      "Parton Level", fancy, mode + "/top-parton"  , plt)
    topc  = top_decay_stats(children,  "c_ntops",  "c_ltops",  "c_htops", "TopChildren Level", fancy, mode + "/top-children", plt)
    toptj = top_decay_stats(truthjet, "tj_ntops", "tj_ltops", "tj_htops",   "TruthJets Level", fancy, mode + "/truth-jets"  , plt)

    #top_mass_dist(tops    ,  "p",      "Parton Level", fancy, mode + "/top-parton"  , plt, 150, 200,  50,  5.0)
    #top_mass_dist(children,  "c", "TopChildren Level", fancy, mode + "/top-children", plt, 100, 250, 150, 10.0)
    top_mass_dist(truthjet, "tj",   "TruthJets Level", fancy, mode + "/truth-jets"  , plt,   0, 300, 300, 10.0)

    topc_cn  = constrain_top_mass(children,  "c", "TopChildren Level", fancy, mode + "/top-children", plt, 100, 300, 200, 10.0, enforce_sym)
    toptj_cn = constrain_top_mass(truthjet, "tj",   "TruthJets Level", fancy, mode + "/truth-jets"  , plt, 100, 300, 200, 10.0, enforce_sym_tj)

    toptj_nj = constrain_top_njets(truthjet, "tj", "TruthJets Level", fancy, mode + "/truth-jets"   , plt, 0, 400, 200, 20.0)

    print("========================== (" + fancy + ") =========================")
    Tabular(topx,    None,  "Truth Tops"              , "Truth Tops")
    Tabular(topx,    topc, "TopChildren"              , "TopChildren")
    Tabular(topx, topc_cn, "TopChildren (Constrained)", "TopChildren")
  
    Tabular(topx,        toptj, "TruthJets (Unconstrained)"  , "TruthJets")
    Tabular(topx,     toptj_cn, "TruthJets (Constrained)"    , "TruthJets")
    TabularFlow(topx, toptj_nj, "TruthJets (Over/Under flow)", "TruthJets", 0, 400)




def entry(smpls, path):
    if "EXP_MC20" in smpls: entry_point("Experimental MC20", "experimental-mc20", path + "experimental_mc20/*", "nominal_Loose", True)


