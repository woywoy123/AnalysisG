def mapping(name):
    if "_singletop_"  in name: return ( "singletop" , "singletop"             )
    if "_ttH125_"     in name: return ( "ttH125"    , "ttH"                   )
    if "_ttbarHT1k_"  in name: return ( "ttbarHT1k" , "$t\\bar{t}$"           )
    if "_SM4topsNLO_" in name: return ( "SM4topsNLO", "$t\\bar{t}t\\bar{t}"   )
    if "_ttbar_"      in name: return ( "ttbar"     , "$t\\bar{t}$"           )
    if "_ttbarHT1k5_" in name: return ( "ttbarHT1k5", "$t\\bar{t}$"           )
    if "_ttbarHT6c_"  in name: return ( "ttbarHT6c" , "$t\\bar{t}$"           )
    if "_Ztautau_"    in name: return ( "Ztautau"   , "$Z\\ell\\ell$"         )
    if "_llll_"       in name: return ( "llll"      , "$\\ell\\ell\\ell\\ell$")
    if "_lllv_"       in name: return ( "lllv"      , "$\\ell\\ell\\ell\\nu$" )
    if "_llvv_"       in name: return ( "llvv"      , "$\\ell\\ell\\nu\\nu$"  )
    if "_lvvv_"       in name: return ( "lvvv"      , "$\\ell\\nu\\nu\\nu$"   )
    if "_tchan_"      in name: return ( "tchan"     , "tchan"                 )
    if "_tt_"         in name: return ( "tt"        , "tt"                    )
    if "_ttee_"       in name: return ( "ttee"      , "tt\\ell\\ell"          )
    if "_ttmumu_"     in name: return ( "ttmumu"    , "tt\\ell\\ell"          )
    if "_tttautau_"   in name: return ( "tttautau"  , "tt\\ell\\ell"          )
    if "_ttW_"        in name: return ( "ttW"       , "ttW"                   )
    if "_ttZnunu_"    in name: return ( "ttZnunu"   , "ttZvv"                 )
    if "_ttZqq_"      in name: return ( "ttZqq"     , "ttZqq"                 )
    if "_tW_"         in name: return ( "tW"        , "tW"                    )
    if "_tZ_"         in name: return ( "tZ"        , "tZ"                    )
    if "_Wenu_"       in name: return ( "Wenu"      , "Wev"                   )
    if "_WH125_"      in name: return ( "WH125"     , "WH"                    )
    if "_WlvZqq_"     in name: return ( "WlvZqq"    , "W\\ell\\nuZqq"         )
    if "_Wmunu_"      in name: return ( "Wmunu"     , "W\\ell\\nu"            )
    if "_WplvWmqq_"   in name: return ( "WplvWmqq"  , "Wp\\ell\\nuWmqq"       )
    if "_WpqqWmlv_"   in name: return ( "WpqqWmlv"  , "WpqqWm\\ell\\nu"       )
    if "_WqqZll_"     in name: return ( "WqqZll"    , "WqqZ\\ell\\ell"        )
    if "_WqqZvv_"     in name: return ( "WqqZvv"    , "WqqZ\\nu\\nu"          )
    if "_Wt_"         in name: return ( "Wt"        , "Wt"                    )
    if "_Wtaunu_"     in name: return ( "Wtaunu"    , "W\\ell\\nu"            )
    if "_Zee_"        in name: return ( "Zee"       , "Z\\ell\\ell"           )
    if "_ZH125_"      in name: return ( "ZH125"     , "ZH"                    )
    if "_Zmumu_"      in name: return ( "Zmumu"     , "Z\\ell\\ell"           )
    if "_ZqqZll_"     in name: return ( "ZqqZll"    , "ZqqZ\\ell\\ell"        )
    if "_ZqqZvv_"     in name: return ( "ZqqZvv"    , "ZqqZ\\nu\\nu"          )
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
