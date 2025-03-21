from AnalysisG.core.plotting import TH1F, TH2F
from helper import *


def entry_point(fancy, mode, pth, tree, plt):

    tops = fetch_data(pth, tree, "top-partons")
    children = fetch_data(pth, tree, "top-children")
    truthjet = fetch_data(pth, tree, "top-truthjet")

    topx  = top_decay_stats(tops    ,  "p_ntops",  "p_ltops",  "p_htops",      "Parton Level", fancy, mode + "/top-parton"  , False)
    topc  = top_decay_stats(children,  "c_ntops",  "c_ltops",  "c_htops", "TopChildren Level", fancy, mode + "/top-children", False)
    toptj = top_decay_stats(truthjet, "tj_ntops", "tj_ltops", "tj_htops",   "TruthJets Level", fancy, mode + "/truth-jets"  , True)

    top_mass_dist(tops    ,  "p_top_mass",  None          ,  None          ,      "Parton Level", fancy, mode + "/top-parton"  , False, 150, 200,  50,  5.0)
    top_mass_dist(children,  "c_top_mass",  "c_isleptonic",  "c_ishadronic", "TopChildren Level", fancy, mode + "/top-children", False, 100, 250, 150, 10.0)
    top_mass_dist(truthjet, "tj_top_mass", "tj_isleptonic", "tj_ishadronic",   "TruthJets Level", fancy, mode + "/truth-jets"  , True ,   0, 300, 300, 10.0)

    exit()
#    constrain_top_mass(children, "c_pdgid", "c_top_mass", "TopChildren Level", fancy, mode + "/top-children", True, 100, 300, 200, 10.0)




    tn_tops, tn_ltop, tn_htop, tn_count = topx
    cn_tops, cn_ltop, cn_htop, cn_count = topc
    tj_tops, tj_ltop, tj_htop, tj_count = toptj

    print("========================== (" + fancy + ") =========================")
    print("---------------------- Truth Tops ------------------------ ")
    print("Initial Number of Tops", tn_tops)
    print("Top-Multiplicity", tn_count)
    print("Leptonic", tn_ltop, " | (%)",  100*tn_ltop / tn_tops)
    print("Hadronic", tn_htop, " | (%)",  100*tn_htop / tn_tops)

    print("---------------------- TopChildren ------------------------ ")
    print("TopChildren Number of Tops", cn_tops, " | Loss (%)", loss(tn_tops, cn_tops))
    print("Top-Multiplicity", cn_count)
    print("Leptonic", cn_ltop, " | Loss (%)", loss(tn_ltop, cn_ltop))
    print("Hadronic", cn_htop, " | Loss (%)", loss(tn_htop, cn_htop))


def entry(smpls, path):
    if "EXP_MC20" in smpls: entry_point("Experimental MC20", "experimental-mc20", path + "experimental_mc20/*", "nominal_Loose", True)


