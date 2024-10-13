def mapping(name):
    if "_singletop_"  in name: return ( "singletop" , "$t$"                      )
    if "_ttH125_"     in name: return ( "ttH"       , "$t\\bar{t}H$"             )
    if "_ttbarHT1k_"  in name: return ( "ttbar"     , "$t\\bar{t}$"              )
    if "_SM4topsNLO"  in name: return ( "SM4topsNLO", "$t\\bar{t}t\\bar{t}$"     )
    if "_ttbar_"      in name: return ( "ttbar"     , "$t\\bar{t}$"              )
    if "_ttbarHT1k5_" in name: return ( "ttbar"     , "$t\\bar{t}$"              )
    if "_ttbarHT6c_"  in name: return ( "ttbar"     , "$t\\bar{t}$"              )
    if "_Ztautau_"    in name: return ( "Zll"       , "$Z\\ell\\ell$"            )
    if "_llll"        in name: return ( "llll"      , "$\\ell\\ell\\ell\\ell$"   )
    if "_lllv"        in name: return ( "lllv"      , "$\\ell\\ell\\ell\\nu$"    )
    if "_llvv"        in name: return ( "llvv"      , "$\\ell\\ell\\nu\\nu$"     )
    if "_lvvv"        in name: return ( "lvvv"      , "$\\ell\\nu\\nu\\nu$"      )
    if "_tchan_"      in name: return ( "tchan"     , "tchan"                    )
    if "_tt_"         in name: return ( "tt"        , "$t\\bar{t}$"              )
    if "_ttee."       in name: return ( "ttll"      , "$t\\bar{t}\\ell\\ell$"    )
    if "_ttmumu."     in name: return ( "ttll"      , "$t\\bar{t}\\ell\\ell$"    )
    if "_tttautau."   in name: return ( "ttll"      , "$t\\bar{t}\\ell\\ell$"    )
    if "_ttW."        in name: return ( "ttW"       , "$t\\bar{t}W$"             )
    if "_ttZnunu."    in name: return ( "ttZll"     , "$t\\bar{t}Z\\nu\\nu$"     )
    if "_ttZqq."      in name: return ( "ttZqq"     , "$t\\bar{t}Zqq$"           )
    if "_tW."         in name: return ( "tW"        , "$tW$"                     )
    if "_tZ."         in name: return ( "tZ"        , "$tZ$"                     )
    if "_Wenu_"       in name: return ( "Wlv"       , "$W\\ell\\nu$"             )
    if "_WH125."      in name: return ( "WH"        , "$WH$"                     )
    if "_WlvZqq"      in name: return ( "WlvZqq"    , "$W\\ell\\nu Zqq$"         )
    if "_Wmunu_"      in name: return ( "Wmunu"     , "$W\\ell\\nu$"             )
    if "_WplvWmqq"    in name: return ( "WplvWmqq"  , "$Wp\\ell\\nu Wmqq$"       )
    if "_WpqqWmlv"    in name: return ( "WpqqWmlv"  , "$WpqqWm\\ell\\nu$"        )
    if "_WqqZll"      in name: return ( "WqqZll"    , "$WqqZ\\ell\\ell$"         )
    if "_WqqZvv"      in name: return ( "WqqZvv"    , "$WqqZ\\nu\\nu$"           )
    if "_Wt_"         in name: return ( "Wt"        , "$Wt$"                     )
    if "_Wtaunu_"     in name: return ( "Wtaunu"    , "$W\\ell\\nu$"             )
    if "_Zee_"        in name: return ( "Zee"       , "$Z\\ell\\ell$"            )
    if "_ZH125_"      in name: return ( "ZH"        , "$ZH$"                     )
    if "_WH125_"      in name: return ( "WH"        , "$WH$"                     )
    if "_Zmumu_"      in name: return ( "Zmumu"     , "$Z\\ell\\ell$"            )
    if "_ZqqZll"      in name: return ( "ZqqZll"    , "$ZqqZ\\ell\\ell$"         )
    if "_ZqqZvv"      in name: return ( "ZqqZvv"    , "$ZqqZ\\nu\\nu$"           )
    if "ttH_tttt"     in name: return ( "tttt_mX"   , "$t\\bar{t}t\\bar{t}H_{X}$")
    print(name)
    exit()
    return "ndef"


def kinesplit(reg):
    x = reg.split(",")
    pt = x[0].split("<")
    eta = x[1].split("<")
    return float(pt[0]), float(pt[2]), float(eta[0]), float(eta[2])

def add_entry(inpt, key, fname, val, weight):
    if key not in inpt: inpt[key] = {}
    if fname not in inpt[key]: inpt[key][fname] = {"value" : [], "weights" : []}
    if not isinstance(val, list): val = [val]; weight = [weight]
    else: weight = [weight]*len(val)
    inpt[key][fname]["value"] += val
    inpt[key][fname]["weights"] += weight

def top_pteta(stacks, data):
    pred_mass = data.p_topmass
    tru_mass  = data.t_topmass
    top_scr   = data.prob_tops

    if "truth" not in stacks: stacks = {"truth" : {}, "prediction" : {}, "top_score" : {}}
    for reg in pred_mass:
        ptl, pth, etl, eth = kinesplit(reg)
        rg = str(int(ptl)) + "_" + str(int(pth)) + ", " + str(etl) + "-" + str(eth)
        hashes = list(pred_mass[reg])
        fn_weight = data.HashToWeightFile(hashes)
        for i in range(len(hashes)):
            hash = hashes[i]
            fn, weight = fn_weight[i]
            fn = "/".join(fn.decode("utf-8").split("/")[-2:])
            add_entry(stacks["prediction"], rg, fn, pred_mass[reg][hash], weight)
            add_entry(stacks["top_score"] , rg, fn,   top_scr[reg][hash], weight)

    for reg in tru_mass:
        ptl, pth, etl, eth = kinesplit(reg)
        rg = str(int(ptl)) + "_" + str(int(pth)) + ", " + str(etl) + "-" + str(eth)
        hashes = list(tru_mass[reg])
        fn_weight = data.HashToWeightFile(hashes)
        for i in range(len(hashes)):
            hash = hashes[i]
            fn, weight = fn_weight[i]
            add_entry(stacks["truth"], rg, fn, tru_mass[reg][hash], weight)
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

def ntops_reco_compl(stacks, data, target):
    n_perfect_tops = data.n_perfect_tops
    n_pred_tops    = data.n_pred_tops
    n_tru_tops     = data.n_tru_tops
    if "e_ntop" not in stacks: stacks |= {"e_ntop" : {}, "p_ntop" : {}, "cls_ntop_w" : {}}
    if target not in stacks["e_ntop"]:
        stacks["e_ntop"][target] = []
        stacks["p_ntop"][target] = []
        stacks["cls_ntop_w"][target] = []

    hashes_ = list(n_perfect_tops)
    fn_weight = data.HashToWeightFile(hashes_)
    for h in range(len(hashes_)):
        hash = hashes_[h]
        fn, weight = fn_weight[h]
        weight = 1
        stacks["cls_ntop_w"][target] += [(n_tru_tops[hash] == target)*weight]
        stacks["e_ntop"][target]     += [(n_perfect_tops[hash] == target)*weight]
        stacks["p_ntop"][target]     += [(n_pred_tops[hash] == target)*weight]
    return stacks
