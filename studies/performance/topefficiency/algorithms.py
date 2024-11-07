def mapping(name):
    if "_singletop_"  in name: return ( "singletop" , "$t$"                      , "aqua")
    if "_tchan_"      in name: return ( "tchan"     , "$t$"                      , "aqua")
    if "_ttbarHT1k_"  in name: return ( "ttbar"     , "$t\\bar{t}$"              , "violet")
    if "_ttbar_"      in name: return ( "ttbar"     , "$t\\bar{t}$"              , "violet")
    if "_ttbarHT1k5_" in name: return ( "ttbar"     , "$t\\bar{t}$"              , "violet")
    if "_ttbarHT6c_"  in name: return ( "ttbar"     , "$t\\bar{t}$"              , "violet")
    if "_tt_"         in name: return ( "tt"        , "$t\\bar{t}$"              , "violet")
    if "_ttee."       in name: return ( "ttll"      , "$t\\bar{t}\\ell\\ell$"    , "orange")
    if "_ttmumu."     in name: return ( "ttll"      , "$t\\bar{t}\\ell\\ell$"    , "orange")
    if "_tttautau."   in name: return ( "ttll"      , "$t\\bar{t}\\ell\\ell$"    , "orange")
    if "_ttW."        in name: return ( "ttW"       , "$t\\bar{t}V$"             , "orchid")
    if "_ttZnunu."    in name: return ( "ttZll"     , "$t\\bar{t}V$"             , "orchid")
    if "_ttZqq."      in name: return ( "ttZqq"     , "$t\\bar{t}V$"             , "orchid")
    if "_ttH125_"     in name: return ( "ttH"       , "$t\\bar{t}H$"             , "blue")
    if "_Wt_"         in name: return ( "Wt"        , "$Wt$"                     , "olive")
    if "_tW."         in name: return ( "tW"        , "$tV$"                     , "teal")
    if "_tW_"         in name: return ( "tW"        , "$tV$"                     , "teal")
    if "_tZ."         in name: return ( "tZ"        , "$tV$"                     , "teal")
    if "_SM4topsNLO"  in name: return ( "SM4topsNLO", "$t\\bar{t}t\\bar{t}$"     , "coral")

    if "_WlvZqq"      in name: return ( "WlvZqq"    , "$WZ$"                     , "darkblue")
    if "_WqqZll"      in name: return ( "WqqZll"    , "$WZ$"                     , "darkblue")
    if "_WqqZvv"      in name: return ( "WqqZvv"    , "$WZ$"                     , "darkblue")
    if "_WplvWmqq"    in name: return ( "WplvWmqq"  , "$WW$"                     , "azure")
    if "_WpqqWmlv"    in name: return ( "WpqqWmlv"  , "$WW$"                     , "azure")
    if "_ZqqZll"      in name: return ( "ZqqZll"    , "$ZZ$"                     , "lime")
    if "_ZqqZvv"      in name: return ( "ZqqZvv"    , "$ZZ$"                     , "lime")
    if "_WH125."      in name: return ( "WH"        , "$VH$"                     , "crimson")
    if "_ZH125_"      in name: return ( "ZH"        , "$VH$"                     , "crimson")
    if "_WH125_"      in name: return ( "WH"        , "$VH$"                     , "crimson")
    if "_Wenu_"       in name: return ( "Wlv"       , "$V\\ell\\nu$"             , "cyan")
    if "_Wmunu_"      in name: return ( "Wmunu"     , "$V\\ell\\nu$"             , "cyan")
    if "_Wtaunu_"     in name: return ( "Wtaunu"    , "$V\\ell\\nu$"             , "cyan")
    if "_Zee_"        in name: return ( "Zee"       , "$V\\ell\\ell$"            , "magenta")
    if "_Zmumu_"      in name: return ( "Zmumu"     , "$V\\ell\\ell$"            , "magenta")
    if "_Ztautau_"    in name: return ( "Zll"       , "$V\\ell\\ell$"            , "magneta")
    if "_llll"        in name: return ( "llll"      , "$\\ell\\ell\\ell\\ell$"   , "green")
    if "_lllv"        in name: return ( "lllv"      , "$\\ell\\ell\\ell\\nu$"    , "grey")
    if "_llvv"        in name: return ( "llvv"      , "$\\ell\\ell\\nu\\nu$"     , "indigo")
    if "_lvvv"        in name: return ( "lvvv"      , "$\\ell\\nu\\nu\\nu$"      , "lightblue")

    if "ttH_tttt" in name:
        m = [str(i*100) for i in range(1, 11) if str(i*100) in name][0]
        return ( "tttt_m"+m, "$t\\bar{t}t\\bar{t}H_{" + m + "}$", "darkgreen")

    print("----> " + name)
    exit()

def kinesplit(reg):
    x = reg.split(",")
    pt = x[0].split("<")
    eta = x[1].split("<")
    return float(pt[0]), float(pt[2]), float(eta[0]), float(eta[2])

def top_pteta(stacks, data, metal):
    pred_mass = data.p_topmass
    tru_mass  = data.t_topmass
    top_scr   = data.prob_tops

    if "truth" not in stacks: stacks = {"truth" : {}, "prediction" : {}, "top_score" : {}}
    mtops_pred, mtops_tru = {}, {}

    for reg in pred_mass:
        ptl, pth, etl, eth = kinesplit(reg)
        rg = str(int(ptl)) + "_" + str(int(pth)) + ", " + str(etl) + "-" + str(eth)
        if rg not in mtops_pred: mtops_pred[rg] = {}

        hashes = list(pred_mass[reg])
        fn_weight = data.HashToWeightFile(hashes)
        for i in range(len(hashes)):
            hash = hashes[i]
            fn, weight = fn_weight[i]
            tops_pred = pred_mass[reg][hash]
            tops_scr  = top_scr[reg][hash]
            weights   = [weight]*len(tops_pred)

            if fn not in mtops_pred[rg]: mtops_pred[rg][fn] = {"w" : [], "m" : [], "s" : []}
            mtops_pred[rg][fn]["w"] += weights
            mtops_pred[rg][fn]["m"] += tops_pred
            mtops_pred[rg][fn]["s"] += tops_scr

    for rg in mtops_pred:
        if rg not in stacks["prediction"]:
            stacks["prediction"][rg] = {}
            stacks["top_score"][rg]  = {}

        for fn in mtops_pred[rg]:
            if fn not in stacks["prediction"][rg]:
                stacks["prediction"][rg][fn] = metal.GenerateData
                stacks["top_score"][rg][fn]  = metal.GenerateData

            stacks["prediction"][rg][fn].data    = {fn : mtops_pred[rg][fn]["m"]}
            stacks["prediction"][rg][fn].weights = {fn : mtops_pred[rg][fn]["w"]}

            stacks["top_score"][rg][fn].data    = {fn : mtops_pred[rg][fn]["s"]}
            stacks["top_score"][rg][fn].weights = {fn : mtops_pred[rg][fn]["w"]}

    for reg in tru_mass:
        ptl, pth, etl, eth = kinesplit(reg)
        rg = str(int(ptl)) + "_" + str(int(pth)) + ", " + str(etl) + "-" + str(eth)
        if rg not in mtops_tru: mtops_tru[rg] = {}

        hashes = list(tru_mass[reg])
        fn_weight = data.HashToWeightFile(hashes)
        for i in range(len(hashes)):
            fn, weight = fn_weight[i]
            hash = hashes[i]
            tops_tru = tru_mass[reg][hash]
            weights  = [weight]*len(tops_tru)
            if fn not in mtops_tru[rg]: mtops_tru[rg][fn] = {"w" : [], "m" : []}
            mtops_tru[rg][fn]["w"] += weights
            mtops_tru[rg][fn]["m"] += tops_tru

    for rg in mtops_tru:
        if rg not in stacks["truth"]: stacks["truth"][rg] = {}
        for fn in mtops_tru[rg]:
            if fn not in stacks["truth"][rg]: stacks["truth"][rg][fn]  = metal.GenerateData
            stacks["truth"][rg][fn].data    = {fn : mtops_tru[rg][fn]["m"]}
            stacks["truth"][rg][fn].weights = {fn : mtops_tru[rg][fn]["w"]}
    return stacks

def roc_data_get(stacks, data):
    if "n-tops_t" not in stacks:
        stacks = {"n-tops_t" : [], "n-tops_p" : [], "signal_t" : [], "signal_p" : [], "edge_top_t" : [], "edge_top_p" : []}

    stacks["n-tops_t"] += data.truth_ntops
    stacks["n-tops_p"] += data.pred_ntops_score

    stacks["signal_t"] += data.truth_signal
    stacks["signal_p"] += data.pred_signal_score

    stacks["edge_top_t"] += data.truth_top_edge
    stacks["edge_top_p"] += data.pred_top_edge_score
    return stacks

def ntops_reco_compl(stacks, data):
    n_perfect_tops = data.n_perfect_tops
    n_pred_tops    = data.n_pred_tops
    n_tru_tops     = data.n_tru_tops
    if "tru_ntops" not in stacks: stacks = {"tru_ntops" : {}, "weights" : {}, "pred_ntops" : {}, "perf_ntops" : {}}

    hashes_ = list(n_perfect_tops)
    fn_weight = data.HashToWeightFile(hashes_)
    for h in range(len(hashes_)):
        fn, weight = fn_weight[h]
        if fn not in stacks["tru_ntops"]:
            stacks["tru_ntops"][fn] = []
            stacks["weights"][fn] = []
            stacks["pred_ntops"][fn] = {}
            stacks["perf_ntops"][fn] = {}

        hash = hashes_[h]
        stacks["weights"][fn].append(weight)
        stacks["tru_ntops"][fn].append(n_tru_tops[hash])
        for score in n_perfect_tops[hash]:
            score_ = round(score, 3)
            if score_ not in stacks["pred_ntops"][fn]: stacks["pred_ntops"][fn][score_] = []
            stacks["pred_ntops"][fn][score_].append(n_pred_tops[hash][score])

            if score_ not in stacks["perf_ntops"][fn]: stacks["perf_ntops"][fn][score_] = []
            stacks["perf_ntops"][fn][score_].append(n_perfect_tops[hash][score])
    return stacks
