def mapping(name):
    if "_singletop_"  in name: return ( "singletop" , "$t$"                      )
    if "_ttH125_"     in name: return ( "ttH"       , "$t\\bar{t}H$"             )
    if "_ttbarHT1k_"  in name: return ( "ttbar"     , "$t\\bar{t}$"              )
    if "_SM4topsNLO"  in name: return ( "SM4topsNLO", "$t\\bar{t}t\\bar{t}"      )
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

def top_pteta(stacks, data):
    pred_mass = data.p_topmass
    tru_mass  = data.t_topmass
    top_scr   = data.prob_tops

    if "truth" not in stacks: stacks = {"truth" : {}, "prediction" : {}, "top_score" : {}}
    for reg in pred_mass:
        for fname in pred_mass[reg]:
            prc, ttl = mapping(fname)
            fn = prc + "#" + ttl
            ptl, pth, etl, eth = kinesplit(reg)

            rg = str(int(ptl)) + "_" + str(int(pth)) + ", " + str(etl) + "-" + str(eth)
            if rg not in stacks["prediction"]: stacks["prediction"][rg] = {}
            if rg not in stacks["top_score"]:  stacks["top_score"][rg] = {}

            if fn not in stacks["prediction"][rg]: stacks["prediction"][rg][fn] = []
            if fn not in stacks["top_score"][rg]:  stacks["top_score"][rg][fn] = []
            stacks["prediction"][rg][fn] += pred_mass[reg][fname]
            stacks["top_score"][rg][fn]  += top_scr[reg][fname]

    for reg in tru_mass:
        for fname in tru_mass[reg]:
            ptl, pth, etl, eth = kinesplit(reg)
            rg = str(int(ptl)) + "_" + str(int(pth)) + ", " + str(etl) + "-" + str(eth)
            if rg not in stacks["truth"]: stacks["truth"][rg] = []
            stacks["truth"][rg] += tru_mass[reg][fname]
    return stacks
