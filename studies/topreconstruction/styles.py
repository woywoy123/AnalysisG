def colors():
    return iter(sorted(list(set([
        "aqua", "orange", "green","blue","olive","teal","gold",
        "darkblue","lime","crimson","magenta","orchid",
        "sienna","salmon","chocolate", "navy", "plum", "indigo", 
    ]))))


def apply_style(hist, title, min_, max_, xtitle, ytitle):
    hist.UseLateX = False
    hist.Style = "ATLAS"
    hist.OutputDirectory = "./output"
    hist.xTitle = xtitle
    hist.yTitle = ytite
    hist.xMin   = min_
    hist.xMax   = max_
    hist.Stacked   = False
    hist.Overflow  = False
    hist.ShowCount = True
    return hist

def mapping(name):
    if "_singletop_"  in name: return ( "singletop" , "$t$"                      , "aqua")
    if "_tchan_"      in name: return ( "tchan"     , "$t$"                      , "aqua")
    if "_ttbarHT1k_"  in name: return ( "ttbar"     , "$t\\bar{t}$"              , "orange")
    if "_ttbar_"      in name: return ( "ttbar"     , "$t\\bar{t}$"              , "orange")
    if "_ttbarHT1k5_" in name: return ( "ttbar"     , "$t\\bar{t}$"              , "orange")
    if "_ttbarHT6c_"  in name: return ( "ttbar"     , "$t\\bar{t}$"              , "orange")
    if "_tt_"         in name: return ( "tt"        , "$t\\bar{t}$"              , "orange")
    if "_ttee."       in name: return ( "ttll"      , "$t\\bar{t}\\ell\\ell$"    , "orange")
    if "_ttmumu."     in name: return ( "ttll"      , "$t\\bar{t}\\ell\\ell$"    , "orange")
    if "_tttautau."   in name: return ( "ttll"      , "$t\\bar{t}\\ell\\ell$"    , "orange")
    if "_ttW."        in name: return ( "ttW"       , "$t\\bar{t}V$"             , "green")
    if "_ttZnunu."    in name: return ( "ttZll"     , "$t\\bar{t}V$"             , "green")
    if "_ttZqq."      in name: return ( "ttZqq"     , "$t\\bar{t}V$"             , "green")
    if "_ttH125_"     in name: return ( "ttH"       , "$t\\bar{t}H$"             , "blue")
    if "_Wt_"         in name: return ( "Wt"        , "$Wt$"                     , "olive")
    if "_tW."         in name: return ( "tW"        , "$tV$"                     , "teal")
    if "_tW_"         in name: return ( "tW"        , "$tV$"                     , "teal")
    if "_tZ."         in name: return ( "tZ"        , "$tV$"                     , "teal")
    if "_SM4topsNLO"  in name: return ( "SM4topsNLO", "$t\\bar{t}t\\bar{t}$"     , "gold")

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
    if "_Ztautau_"    in name: return ( "Zll"       , "$V\\ell\\ell$"            , "magenta")
    if "_llll"        in name: return ( "llll"      , "$\\ell\\ell\\ell\\ell$"   , "orchid")
    if "_lllv"        in name: return ( "lllv"      , "$\\ell\\ell\\ell\\nu$"    , "sienna")
    if "_llvv"        in name: return ( "llvv"      , "$\\ell\\ell\\nu\\nu$"     , "silver")
    if "_lvvv"        in name: return ( "lvvv"      , "$\\ell\\nu\\nu\\nu$"      , "salmon")


