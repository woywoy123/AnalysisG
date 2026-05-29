def mapping(name):
    for i in range(7):
         s = "m" + str(int(i * 100 + 400))
         if s not in name: continue
         return s
   
    i = name
    if "Ztautau"  in i: return "Zll"
    if "Zmumu"    in i: return "Zll"
    if "Zee"      in i: return "Zll"

    if "Wtaunu"   in i: return "Wlnu"
    if "Wmunu"    in i: return "Wlnu"
    if "Wenu"     in i: return "Wlnu"

    if "SM4tops"  in i: return "tttt"

    if "tttautau" in i: return "ttll"
    if "ttmumu"   in i: return "ttll"
    if "ttee"     in i: return "ttll"

    if "ttZqq"    in i: return "ttZ -> qq"
    if "ttZnunu"  in i: return "ttZ -> nunu"

    if "ZqqZll"   in i: return "ZZ -> qq,ll"
    if "WqqZll"   in i: return "WZ -> qq,ll"

    if "llll"     in i: return "llll"
    if "lllv"     in i: return "lllnu"
    if "llvv"     in i: return "llnunu"
    
    if "ttH125"   in i: return "ttH -> ll"
    if "ZH125"    in i: return "ZH"
    if "WH125"    in i: return "WH"

    if "tchan"    in i: return "tchan-lep"
    if "schan"    in i: return "schan-lep"

    if "ttW" in i: return "ttW"

    if "ttbar"     in i and "SingleLep" in i: return "ttbar -> l"
    if "ttbar"     in i and "dil"       in i: return "ttbar -> ll"

    if "tt"        in i and "SingleLep" in i: return "tt -> l"
    if "tt"        in i and "dil"       in i: return "tt -> ll"
    if "singletop" in i:                      return "singletop"

    if "antitop"  in i: return "tbar"
    if "ttbar"    in i: return "ttbar"
    if "tt"       in i: return "tt"
    if "Wt"       in i: return "tW"
    if "tW"       in i: return "tW"
    if "tZ"       in i: return "tZ"
