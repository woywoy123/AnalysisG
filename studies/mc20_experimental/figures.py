from AnalysisG.core.plotting import TH1F, TH2F
from helper import *
import pathlib
import pickle

def null_cut(p, m, pdg, is_l, nj): return True
def enforce_sym(p, m, pgd, is_l, nj):
    if "gamma" in p: return False
    return True

def enforce_sym_tj(p, m, pgd, is_l, nj):
    absx = [abs(i) for i in pgd]
    if len(absx) <= 2: return False
    #if is_l and "o" in p: return False
    #if is_l and absx.count(22) > 1: return False
    #if is_l and "nu" in p and "2" in p: return False
    #if not is_l and absx.count(22) > 3: return False
    #if not is_l and absx.count(11): return False 
    #if not is_l and absx.count(12): return False 
    #if not is_l and absx.count(13): return False 
    #if not is_l and absx.count(14): return False 
    #if not is_l and absx.count(15): return False 
    #if not is_l and absx.count(16): return False 
    #if not is_l and len([i for i in set(absx) if i < 23]) < 2: return False
#    if not is_l: print(pgd)
    return True

def enforce_sym_jet(p, m, pgd, is_l, nj):
    return nj == 1 if is_l else nj == 3

def Tabular(topx, target, title, mode, rec = None):
    print("---------------------- " + title + " ------------------------ ")
    if target is None: 
        print(mag("Top-Multiplicity: "      , 16) + mrg(topx["counts"], 8))
        print(mag("Initial Number of Tops: ", 16) + mrg(topx["ntops"] , 8))
    else: 
        if "counts" in target: print(mag("Top-Multiplicity: ", 16) + mrg(target["counts"]))
        print(mag("Number of Tops: ", 16) + mrg(target["ntops"], 0) + " | Loss (%): " + loss(topx["ntops"], target["ntops"], 4))

    if target is None:
        print(mag("Leptonic: ", 16) + mrg(topx["leptonic"], 0) + " | (%): " +  pc(topx["ntops"], topx["leptonic"], 4))
        print(mag("Hadronic: ", 16) + mrg(topx["hadronic"], 0) + " | (%): " +  pc(topx["ntops"], topx["hadronic"], 4))
        return 
    print(mag("Leptonic: ", 16) + mrg(target["leptonic"], 0) + " | Loss (%): " + loss(topx["leptonic"], target["leptonic"], 4))
    print(mag("Hadronic: ", 16) + mrg(target["hadronic"], 0) + " | Loss (%): " + loss(topx["hadronic"], target["hadronic"], 4))

    if "stats-all" not in target: return
    a_tops ,  o_tops,   u_tops  = (sum(target["stats-all"][k].values()) for k in ["domain", "over", "under"])
    l_tops,  ol_tops,  ul_tops  = (target["stats-all"][k]["leptonic"] for k in ["domain", "over", "under"])
    h_tops,  oh_tops,  uh_tops  = (target["stats-all"][k]["hadronic"] for k in ["domain", "over", "under"])

    p = 6
    at_tops = a_tops + o_tops + u_tops
    al_tops = l_tops + ol_tops + ul_tops
    ah_tops = h_tops + oh_tops + uh_tops

    print("____________________ Raw Values ____________________")
    print(mag("Leptonically Matched: ", 25) + mrg(al_tops, 0) + " | Overflow: " + mrg(ol_tops, 0) + " Underflow: " + mrg(ul_tops, 0))
    print(mag("Hadronically Matched: ", 25) + mrg(ah_tops, 0) + " | Overflow: " + mrg(oh_tops, 0) + " Underflow: " + mrg(uh_tops, 0))
    print(mag("False Assigned (wrong/(lep+had)): ", 16) + mrg(target["wrong"], 0) + " | False Matching (lepton != hadron) (%): " + pc(target["hadronic"] + target["leptonic"], target["wrong"], p))
    print("____________________ Percentage ____________________")
    print(mag("Leptonically (lep/all_leps): ", 30) + pc(al_tops, l_tops, p) + " | Overflow: " + pc(al_tops, ol_tops, p) + " Underflow: " + pc(al_tops, ul_tops, p))
    print(mag("Hadronically (had/all_hads): ", 30) + pc(ah_tops, h_tops, p) + " | Overflow: " + pc(ah_tops, oh_tops, p) + " Underflow: " + pc(ah_tops, uh_tops, p))
    print("")


def TabularFlow(topx, stat, title, mode, min_, max_):
    a_tops,  o_tops,  u_tops = (sum(stat["tops-merged"][k]["all"].values())      for k in ["domain", "over", "under"])
    l_tops, ol_tops, ul_tops = (sum(stat["tops-merged"][k]["leptonic"].values()) for k in ["domain", "over", "under"])
    h_tops, oh_tops, uh_tops = (sum(stat["tops-merged"][k]["hadronic"].values()) for k in ["domain", "over", "under"])

    at_tops = a_tops + o_tops + u_tops
    al_tops = l_tops + ol_tops + ul_tops
    ah_tops = h_tops + oh_tops + uh_tops

    print("---------------------- " + title + " ------------------------ ")
    print("____________________ Raw Values ____________________")
    print(mag(mode + " Number of Tops (" + str(min_) + " -> " + str(max_) + "): ", 30))
    print(mag("All: "                 , 25) + mrg(a_tops, 0) + " | Overflow: " + mrg(o_tops , 0) + " Underflow: " + mrg( u_tops, 0))
    print(mag("Leptonically Matched: ", 25) + mrg(l_tops, 0) + " | Overflow: " + mrg(ol_tops, 0) + " Underflow: " + mrg(ul_tops, 0))
    print(mag("Hadronically Matched: ", 25) + mrg(h_tops, 0) + " | Overflow: " + mrg(oh_tops, 0) + " Underflow: " + mrg(uh_tops, 0))

    p = 6
    print("____________________ Percentage (based on available tops) ____________________")
    print(mag("All Matched: "                , 30) + pc(at_tops, a_tops, p) + " | Overflow: " + pc(at_tops, o_tops , p) + " Underflow: " + pc(at_tops,  u_tops, p))
    print(mag("Leptonically (lep/all_leps): ", 30) + pc(al_tops, l_tops, p) + " | Overflow: " + pc(al_tops, ol_tops, p) + " Underflow: " + pc(al_tops, ul_tops, p))
    print(mag("Hadronically (had/all_hads): ", 30) + pc(ah_tops, h_tops, p) + " | Overflow: " + pc(ah_tops, oh_tops, p) + " Underflow: " + pc(ah_tops, uh_tops, p))

    print("____________________ Loss (relative to initial tops)____________________")
    print(mag("All: ", 6) + loss(topx["ntops"], a_tops, p) + " | Leptonically Matched: " + loss(topx["leptonic"], l_tops, p) + " | Hadronically Matched: " + loss(topx["hadronic"], h_tops, p))
    print("")

def entry_point(fancy, mode, pth, tree, plt = False):

    smplx = [
            "top-partons"      , 
            "top-children"     , 
            "top-truthjet"     , 
            "top-jets-children", 
            "top-jets-leptons"  
    ]

    fnc = ["Parton", "TopChildren", "TruthJet", "Jets Children", "Jets Leptons"]
#    fnc = [i + " Level" for i in fnc]

    pths = ["top-parton", "top-children", "truth-jets", "jets-children", "jets-leptons"]
    pths = [mode + "/" + i for i in pths]

    data     = fetch_all(pth, tree, smplx)
    tops     = fetch_data(data, smplx[0])
    children = fetch_data(data, smplx[1])
    truthjet = fetch_data(data, smplx[2])
    jetschil = fetch_data(data, smplx[3])
    jetslept = fetch_data(data, smplx[4])

    #topx  = top_decay_stats(tops    ,  "p", fnc[0], fancy, pths[0], plt)
    #topc  = top_decay_stats(children,  "c", fnc[1], fancy, pths[1], plt)
    #toptj = top_decay_stats(truthjet, "tj", fnc[2], fancy, pths[2], plt)
    #topjc = top_decay_stats(jetschil, "jc", fnc[3], fancy, pths[3], plt)
    #topjl = top_decay_stats(jetslept, "jl", fnc[4], fancy, pths[4], plt)

    #top_mass_dist(tops    ,  "p", fnc[0], fancy, pths[0], plt, 150, 200, 100,  5.0)
    #top_mass_dist(children,  "c", fnc[1], fancy, pths[1], plt, 140, 200,  60,  5.0)
    #top_mass_dist(truthjet, "tj", fnc[2], fancy, pths[2], plt, 100, 300, 200, 10.0)
    #top_mass_dist(jetschil, "jc", fnc[3], fancy, pths[3], plt, 100, 300, 200, 10.0)
    #top_mass_dist(jetslept, "jl", fnc[4], fancy, pths[4], plt, 100, 300, 200, 10.0)

    #topc_cn  = constrain_top_mass(children,  "c", fnc[1], fancy, pths[1], plt, 140, 200,  60,  5.0, null_cut, ["red", "green", "orange", "purple", "blue", "grey"] )
    #toptj_cn = constrain_top_mass(truthjet, "tj", fnc[2], fancy, pths[2], plt,   0, 300, 300, 40.0, null_cut)
    #topjc_cn = constrain_top_mass(jetschil, "jc", fnc[3], fancy, pths[3], plt,   0, 300, 300, 40.0, null_cut)
    topjl_cn = constrain_top_mass(jetslept, "jl", fnc[4], fancy, pths[4], plt,   0, 300, 300, 40.0, null_cut)
    print("before constraint false matching: ", 100*topjl_cn["wrong"] / (topjl_cn["leptonic"] + topjl_cn["hadronic"]))

    #topc_cn  = constrain_top_mass(children,  "c", fnc[1], fancy, pths[1], plt, 140, 200,  60,  5.0, enforce_sym, ["red", "green", "orange", "purple", "grey"])
    #toptj_cn = constrain_top_mass(truthjet, "tj", fnc[2], fancy, pths[2], plt, 100, 300, 200, 40.0, enforce_sym_tj )
    #topjc_cn = constrain_top_mass(jetschil, "jc", fnc[3], fancy, pths[3], plt, 100, 300, 200, 40.0, enforce_sym_jet)
    topjl_cn = constrain_top_mass(jetslept, "jl", fnc[4], fancy, pths[4], plt, 100, 300, 200, 40.0, enforce_sym_jet)
    print("after containt false matching: ", 100*topjl_cn["wrong"] / (topjl_cn["leptonic"] + topjl_cn["hadronic"]))
    return 
    toptj_nj = constrain_top_njets(truthjet, "tj", fnc[2], fancy, pths[2], plt, 0, 300, 300, 40.0)
    topjc_nj = constrain_top_njets(jetschil, "jc", fnc[3], fancy, pths[3], plt, 0, 300, 300, 40.0)
    topjl_nj = constrain_top_njets(jetslept, "jl", fnc[4], fancy, pths[4], plt, 0, 300, 300, 40.0)

    print("========================== (" + fancy + ") =========================")
    print("====================================================================")
    Tabular(topx,    None,  "Truth Tops"              , "Truth Tops")
    print("")
    print("====================================================================")
    Tabular(topx,    topc, "TopChildren"              , "TopChildren")
    Tabular(topx, topc_cn, "TopChildren (Constrained)", "TopChildren")
    
    print("")
    print("====================================================================")

    Tabular(topx,        toptj, "TruthJets (Unconstrained)"  , "TruthJets")
    Tabular(topx,     toptj_cn, "TruthJets (Constrained)"    , "TruthJets")
    TabularFlow(topx, toptj_nj, "TruthJets (Over/Under flow)", "TruthJets", 0, 400)

    print("")
    print("====================================================================")

    Tabular(topx,        topjc, "Jets Children (Unconstrained)"  , "JetsChildren")
    Tabular(topx,     topjc_cn, "Jets Children (Constrained)"    , "JetsChildren")
    TabularFlow(topx, topjc_nj, "Jets Children (Over/Under flow)", "JetsChildren", 0, 400)
    print("")
    print("====================================================================")

    Tabular(topx,        topjl, "Jets Leptons (Unconstrained)"  , "JetsLeptons")
    Tabular(topx,     topjl_cn, "Jets Leptons (Constrained)"    , "JetsLeptons")
    TabularFlow(topx, topjl_nj, "Jets Leptons (Over/Under flow)", "JetsLeptons", 0, 400)



def entry(smpls, path):
    mode = "dr0.4"
    if "EXP_MC20"  in smpls: entry_point("Fuzzy Matching MC20", "experimental-mc20-" + mode, path + "experimental_mc20/big-" + mode + ".root", "nominal_Loose", True)
    mode = "dr0.2"
    if "EXP_MC20"  in smpls: entry_point("Fuzzy Matching MC20", "experimental-mc20-" + mode, path + "experimental_mc20/big-" + mode + ".root", "nominal_Loose", True)
    if "SSML_MC20" in smpls: entry_point("Top-CP Toolkit MC20", "top-cp-mc20"              , path + "ssml_mc20/*"                            , "reco"         , True)
    if "BSM_4TOPS" in smpls: entry_point("Ghost Matched MC16", "reference-mc16"            , path + "bsm_4tops/*"                            , "nominal"      , True)


