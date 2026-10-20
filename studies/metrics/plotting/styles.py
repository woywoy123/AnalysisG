from AnalysisG.core import TH1F, TH2F
from atomics import Dataloader

from filter import *
from utils import *
from plots import *
import asyncio
import gc
import ctypes

def make_truth(dt):
    stats = stats_t()
    for nxt in dt:
        tru_r, tru_a = nxt["Truth"]["raw"], nxt["Truth"]["adj"]
        for k in tru_a.expand(tru_r): stats.add(k)
           
    raw = stats.TopSummary(None, stats.rtop_train, stats.rtop_valid)
    f = open("meta-data/meta_raw.txt", "w")
    f.write("\n".join([str(stats), "", "", "", raw]))
    f.close()
    
    adj = stats.TopSummary(None, stats.atop_train, stats.atop_valid)
    f = open("meta-data/meta_adj.txt", "w")
    f.write("\n".join([str(stats), "", "", "", adj]))
    f.close()

    prc = {}
    for nxt in dt:
        tru_r, tru_a = nxt["Truth"]["raw"], nxt["Truth"]["adj"]
        for k in tru_a.expand(tru_r):
            if k.prc not in prc: prc[k.prc] = {"raw" : [], "adj" : []}
            prc[k.prc]["raw"] += [t for t in k.truth]
            prc[k.prc]["adj"] += [t for t in k.masses]
    
    # ------------------ Raw excluding signal --------------------# 
    bgk = [basic_th1f(kx, prc[kx]["raw"]) for kx in prc if "0)" not in kx]
    th1 = templ_th1f("", 400, 0, 400, [], "Invariant Mass of Truth Tops (GeV)", 
            "Number of Tops (Arb.) / (1 GeV)", sort_hist(bgk), colors()
    )
    th1.xStep           = 20
    th1.Stacked         = True
    th1.Filename        = "background_truth_tops_raw"
    th1.OutputDirectory = "./Plots/TruthInformation/"
    th1.SaveFigure()
  
    # ------------------ Raw including signal --------------------# 
    bgk = [basic_th1f(kx, prc[kx]["raw"]) for kx in prc if "0)" in kx]
    th1 = templ_th1f("", 400, 0, 400, [], "Invariant Mass of Truth Tops (GeV)", 
            "Number of Tops (Arb.) / (1 GeV)", sort_hist(bgk), colors()
    )
    th1.xStep           = 20
    th1.Stacked         = True
    th1.Filename        = "signal_truth_tops_raw"
    th1.OutputDirectory = "./Plots/TruthInformation/"
    th1.SaveFigure()
   
    # ------------------ Adj excluding signal --------------------# 
    bgk = [basic_th1f(kx, prc[kx]["adj"]) for kx in prc if "0)" not in kx]
    th1 = templ_th1f("", 400, 0, 400, [], "Invariant Mass of Truth Tops (GeV)", 
            "Number of Tops (Arb.) / (1 GeV)", sort_hist(bgk), colors()
    )
    th1.xStep           = 20
    th1.Stacked         = True
    th1.Filename        = "background_truth_tops_adj"
    th1.OutputDirectory = "./Plots/TruthInformation/"
    th1.SaveFigure()
  
    # ------------------ Raw including signal --------------------# 
    bgk = [basic_th1f(kx, prc[kx]["adj"]) for kx in prc if "0)" in kx]
    th1 = templ_th1f("", 400, 0, 400, [], "Invariant Mass of Truth Tops (GeV)", 
            "Number of Tops (Arb.) / (1 GeV)", sort_hist(bgk), colors()
    )
    th1.xStep           = 20
    th1.Stacked         = True
    th1.Filename        = "signal_truth_tops_adj"
    th1.OutputDirectory = "./Plots/TruthInformation/"
    th1.SaveFigure()
   
def make_reco(dt):
    for j in dt:

        tru_r, tru_a = j["Truth"]["raw"]   , j["Truth"]["adj"]
        name = ".".join([tru_r.model, tru_r.epoch, tru_r.kfold])

        msk_r, msk_a = j["Masked"]["raw"]  , j["Masked"]["adj"]
        msk_R, msk_A = msk_r.expand(tru_r), msk_a.expand(tru_a)
        
        make_sequence(msk_R, "raw_mask", name, 1.0)
        make_sequence(msk_A, "adj_mask", name, 1.0)

        usk_r, usk_a = j["Unmasked"]["raw"], j["Unmasked"]["adj"]
        usk_R, usk_A = usk_r.expand(tru_r), usk_a.expand(tru_a)

        make_sequence(usk_R, "raw_unmasked", name, 1.0)
        make_sequence(usk_A, "adj_unmasked", name, 1.0)

        nom_r, nom_a = j["Nominal"]["raw"] , j["Nominal"]["adj"]
        nom_R, nom_A = nom_r.expand(nom_r), nom_a.expand(nom_a)

        make_sequence(nom_R, "raw_nominal", name, 12.0)
        make_sequence(nom_A, "adj_nominal", name, 12.0)

        del j, tru_r, tru_a, msk_r, msk_a, usk_r, usk_a, 
        del nom_r, nom_a, msk_R, msk_A, 
        del usk_R, usk_A, nom_R, nom_A
        gc.collect()
        continue

        gm, gw = prc["gm"], prc["gw"]
        th = basic_th2f("", gm, gw)
        default(th)
        th.xTitle = "Invariant Mass of Candidate Tops (GeV) / ($2$ GeV)"
        th.xStep  = 40
        th.xMin   = 0 
        th.xMax   = 400
        th.xBins  = 200

        th.yTitle = "Top Candidate Score (" + nm + ")"
        th.yStep  = 1 if rlx > 1 else 0.1
        th.yMin   = 0 
        th.yMax   = rlx
        th.yBins  = 100
        th.Color  = "magma"

        th.Filename = "score_map_" + nm.lower()
        th.OutputDirectory = "./Plots/" + name.replace(".", "/")
        th.SaveFigure()


def make_perf(dt):

    x = 0
    prf_r = performance_t()
    for j in dt:
        tru_r, tru_a = j["Truth"]["raw"]   , j["Truth"]["adj"]
        msk_r, msk_a = j["Masked"]["raw"]  , j["Masked"]["adj"]
        usk_r, usk_a = j["Unmasked"]["raw"], j["Unmasked"]["adj"]
        nom_r, nom_a = j["Nominal"]["raw"] , j["Nominal"]["adj"]

        prf_r.evaluate(msk_r, tru_r, "(Raw.) PageRank MASKED"  )
        prf_r.evaluate(usk_r, tru_r, "(Raw.) PageRank UNMASKED")
        prf_r.evaluate(nom_r, tru_r, "(Raw.) NOMINAL" )
        print("=================================")

        prf_r.evaluate(msk_a, tru_a, "(Adj.) PageRank MASKED"  )
        prf_r.evaluate(usk_a, tru_a, "(Adj.) PageRank UNMASKED")
        prf_r.evaluate(nom_a, tru_a, "(Adj.) NOMINAL" )
        print("=================================")
        print("---->", tru_r.epoch, tru_r.kfold, tru_r.model)
        del j, tru_r, tru_a, msk_r, msk_a, usk_r, usk_a, nom_r, nom_a
        gc.collect()
        if x % 10 == 9: prf_r.summary()
        

    prf_r.summary()
    prf_r.save()
    return prf_r


def entry(mdl, kfold, epoch, base_dir):
    #make_truth(Dataloader(["MRK-1"], kfold, ["epoch-1"], base_dir))
#    make_reco(Dataloader(mdl, kfold, epoch, base_dir))
    sm = make_perf(Dataloader(mdl, kfold, epoch, base_dir))
 

