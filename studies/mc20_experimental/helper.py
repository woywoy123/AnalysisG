from AnalysisG.core.plotting import TH1F, TH2F
from AnalysisG.core.io import IO
import tqdm

def mag(p, m = 8): return "".join([" "]*(m - len(str(p)))) + p

def mrg(p, m = 7): 
    strx = ""
    try: strx = str("%." + str(m-2) + "f") % p
    except: strx = str(p)
    if "." not in strx: l = 0
    else: l = len(strx.split(".")[0])
    return "".join([" "]*(m - len(strx))) + str(strx)

def colors(cl = ["red", "green", "orange", "purple", "pink"]): return iter(cl)
def count(data): return {c : sum([c == t for t in data]) for c in list(set(data))}

def countv(data, force = False):  
    if not force: return {c : {j : len(data[c][j]) for j in data[c]} for c in data}
    try: return {c : len(data[c]) for c in data}
    except: return len(data)

def toGeV(data): return [x / 1000.0 for x in data]
def loss(t,  f, dm = None): 
    if t == 0: return mrg("NAN", 4)
    lx = ((t - f) / t)*100.0
    if lx < 0: lx = 100 + abs(lx)
    if dm is None: return lx
    return mrg(round(lx, dm), dm + 5)

def pc(t, f, m = None): 
    if t == 0: return mrg("NAN") if m is None else mrg("NAN", m)
    return mrg(100*f / t) if m is None else mrg(100 * f / t, m)

def _title(level, fancy): return level + "\n (" + fancy + ")"
def safe(data, key):
    try: return data[key]
    except: return None

def make(data, prx, key):
    try: return data[prx + "_" + key]
    except: return None

def bar(p, sym): return sym if p > 0 else "\\bar{" + sym + "}"
def sorted_index(seq): return [i[0] for i in sorted(enumerate(seq), key=lambda x:x[1])]

def njets_format(data):
    xs = "Jets"
    if data >= 4: return "\\geq \\texttt{4-" + xs + "}"
    return "\\texttt{" + str(data) + "-" + xs + "}"

def ntops_format(data): return "\\texttt{" + str(max(data + [0])) + "-Tops}"
    
def get_overflow(data, max_, min_, nst = False):
    if not nst:
        overflw = {i : {j : [] for j in data[i]} for i in data}
        undrflw = {i : {j : [] for j in data[i]} for i in data}
        for key in data:
            for kx in data[key]: 
                overflw[key][kx] = [v for v in data[key][kx] if v >= max_]
                undrflw[key][kx] = [v for v in data[key][kx] if v <= min_]
                data[key][kx]    = [v for v in data[key][kx] if v > min_ and v < max_]
        return undrflw, overflw
        
    overflw = {i : [] for i in data}
    undrflw = {i : [] for i in data}
    for key in data:
        overflw[key] = [v for v in data[key] if v >= max_]
        undrflw[key] = [v for v in data[key] if v <= min_]
        data[key]    = [v for v in data[key] if v > min_ and v < max_]
    return undrflw, overflw

def symbolic(p):
    if isinstance(p, list): 
        lx = [symbolic(p[k]) for k in sorted_index([i for i in p])]
        xol = {l : [l for k in lx if k == l] for l in set(lx)}
        srt = sorted(xol)
        return ", ".join([l + "^{" + str(len(xol[l])) + "}" for l in srt])

    if abs(p) == 1:  return bar(abs(p), "d")
    if abs(p) == 2:  return bar(abs(p), "u")
    if abs(p) == 3:  return bar(abs(p), "s")
    if abs(p) == 4:  return bar(abs(p), "c")
    if abs(p) == 5:  return bar(abs(p), "b")
    if abs(p) == 11: return bar(abs(p), "e")
    if abs(p) == 12: return bar(abs(p), "\\nu_{e}")
    if abs(p) == 13: return bar(abs(p), "\\mu")
    if abs(p) == 14: return bar(abs(p), "\\nu_{\\mu}")
    if abs(p) == 15: return bar(abs(p), "\\tau")
    if abs(p) == 16: return bar(abs(p), "\\nu_{\\tau}")
    if abs(p) == 21: return bar(abs(p), "g")
    if abs(p) == 22: return bar(abs(p), "\\gamma")
    if abs(p) == 25: return bar(abs(p), "H")
    if abs(p) == 33: return "Z (33)"
    return "o"

def default(hist, pth):
    hist.Style = r"ATLAS"
    if pth is not None: hist.OutputDirectory = "figures/" + pth
    hist.DPI = 250
    hist.TitleSize = 20
    hist.AutoScaling = True
    hist.Overflow = False
    hist.yScaling = 10
    hist.xScaling = 15
    hist.FontSize = 15
    hist.AxisSize = 14
    return hist

def template(xdata, title, color = None, pth = None, params = {}):
    thl = default(TH1F(), pth)
    if xdata is not None: thl.xData = xdata
    if color is not None: thl.Color = color
    thl.xMin = 0
    thl.Alpha = 0.5
    thl.Title = title
    for i in params: setattr(thl, i, params[i])
    return thl

def top_decay_stats(data, prefix, level, fancy, mode, plt):
    num_tops , num_ltop ,  num_htop = make(data, prefix, "ntops"), make(data, prefix, "ltops"), make(data, prefix, "htops")
    init_tops, init_ltop, init_htop = sum(num_tops), sum(num_ltop), sum(num_htop)
    if not plt: return {"ntops" : init_tops, "leptonic" : init_ltop, "hadronic": init_htop, "counts": count(num_tops)}

    thl = template(num_tops, "Top Multiplicity at " + _title(level, fancy), "blue", mode)
    thl.xTitle = "Number of Tops Per Event"
    thl.yTitle = "Number of Unweighted Events"
    thl.xBins = 6
    thl.xStep = 1
    thl.xMin = 0
    thl.xMax = 6
    thl.Filename = "num-tops"
    thl.dump()
    thl.SaveFigure()

    thl = template(num_ltop, "Leptonic", "blue")
    thl.IntegratedLuminosity = (init_ltop / init_tops)
    thl.CrossSection = 100.0
    thl.ApplyScaling = True
    thl.Density = True

    thh = template(num_htop, "Hadronic", "red")
    thh.CrossSection = 100.0
    thh.IntegratedLuminosity = (init_htop / init_tops)
    thh.ApplyScaling = True
    thh.Density = True
    thh.Hatch = "-"

    tpl = template(None, "Top Decay Mode Event Multiplicity Composition at " + _title(level, fancy), None, mode)
    tpl.Histograms = [thl, thh] 
    tpl.xTitle = "Number of Tops / Per Event"
    tpl.yTitle = "Percentage of Events (%)"
    tpl.xMax = 5
    tpl.xBins = 5
    tpl.xStep = 1
    tpl.Filename = "top-decay-mode"
    tpl.dump()
    tpl.SaveFigure()
    return {"ntops" : init_tops, "leptonic" : init_ltop, "hadronic": init_htop, "counts": count(num_tops)}

def top_mass_dist(data, prefix, level, fancy, mode, plt, _min, _max, _bins, _step):
    ml_top = make(data, prefix if not "p" else None, "isleptonic")
    mh_top = make(data, prefix if not "p" else None, "ishadronic")
    m_top  = make(data, prefix, "top_mass")
    if not plt: return 

    print("----------")
    m_top = [j for i in m_top for j in i]
    if ml_top is not None: ml_top = [j for i in ml_top for j in i]
    else: ml_top = []

    if mh_top is not None: mh_top = [j for i in mh_top for j in i]
    else: mh_top = []
    print("----------")

    _mode = {"All" : [], "Hadronic" : [], "Leptonic" : [], "Wrong" : [], "Overflow" : [], "Underflow" : []}
    for i in tqdm.trange(len(m_top)):
        mt = m_top[i]
        if mt < _min: _mode["Underflow"].append(mt); continue
        if mt > _max: _mode["Overflow"].append(mt); continue
        _mode["All"].append(mt)

        try: 
            if ml_top[i] != mh_top[i]: pass
            else: _mode["Wrong"].append(mt); continue
        except IndexError: continue
        kx = "Leptonic" if ml else "Hadronic"
        _mode[kx].append(mt)

    hist = []
    if len(mh_top): hist.append(template(_mode["Leptonic"], "Leptonic", "red" , None))
    if len(ml_top): hist.append(template(_mode["Hadronic"], "Hadronic", "blue", None, {"Hatch" : "//////\\\\\\\\\\\\\\"}))
    if len(_mode["Wrong"]): hist.append(template(_mode["Wrong"], "Falsely Matched", "orange" , None))

    thm = template(None, "Invariant Mass Distribution of Matched Tops using " + _title(level, fancy), "blue", mode)
    if len(hist): thm.Histograms = hist
    if not len(hist): thm.xData = _mode["All"]
    thm.Overflow = False
    thm.xTitle = "Top Mass (GeV)"
    thm.yTitle = "Number of Tops / " + str((_max - _min)/_bins) + " GeV"
    thm.xMax = _max
    thm.xMin = _min
    thm.xStep = _step
    thm.xBins = _bins
    thm.Filename = "top-mass"
    thm.dump()
    thm.SaveFigure()

def constrain_top_mass(data, prefix, level, fancy, mode, plt, _min, _max, _bins, _step, constr_fx):
    pdgid, top_m = make(data, prefix,      "pdgid"), make(data, prefix,   "top_mass")
    is_l, is_h   = make(data, prefix, "isleptonic"), make(data, prefix, "ishadronic")
    is_w, is_wt  = make(data, prefix, "num_false" ), make(data, prefix, "wrong_match")

    wrong = {}
    leptonic = {}
    hadronic = {}
    ltop, htop, wx = 0, 0, 0
    for x in range(len(pdgid)):
        is_wrg = 0
        if is_wt is not None: 
            top_m[x] += is_wt[x]
            is_wrg = is_w[x]

        for y in range(len(pdgid[x])):
            sym = symbolic(pdgid[x][y])
            is_l_, is_h_ = is_l[x][y], is_h[x][y]

            if not constr_fx(sym, top_m[x][y], pdgid[x][y]): continue
            if   is_l_ > 0: modes = leptonic; ltop += 1
            elif is_h_ > 0: modes = hadronic; htop += 1
            else: modes = wrong; wx+=1

            if sym not in modes: modes[sym] = []
            modes[sym].append(top_m[x][y])

    lm = [j for k in leptonic for j in leptonic[k]]
    hm = [j for k in hadronic for j in hadronic[k]]


    frl = {k : len(leptonic[k]) / len(lm) for k in  leptonic}
    frh = {k : len(hadronic[k]) / len(hm) for k in  hadronic}
    wrg = {k : len(wrong[k])    / (wx+1) for k in  wrong}


    frl = list(dict(sorted(frl.items(), key=lambda item: item[1], reverse = True)))
    frh = list(dict(sorted(frh.items(), key=lambda item: item[1], reverse = True)))
    wrg = list(dict(sorted(wrg.items(), key=lambda item: item[1], reverse = True)))


    atops = {"all" : hm + lm, "leptonic" : lm, "hadronic" : hm}
    a_udr, a_ovr = get_overflow(atops, _max, _min, True)

    data = {"leptonic" : ltop, "hadronic": htop, "ntops": htop + ltop, "wrong" : wx}
    data |= {"stats-all" : {"over" : countv(a_ovr, True), "under": countv(a_udr, True), "domain" : countv(atops, True)}}
    if not plt: return data

    hist = [template(leptonic[k], r"$" + k + "$", c, None, {"ErrorBars" : False}) for k, c in zip(frl, colors())]
    thm = template(None, "Invariant Mass of Leptonically Matched Tops by Symbolic Decay at " + _title(level, fancy), None, mode)
    thm.Overflow = False
    thm.Stacked = False
    thm.ErrorBars = False
    thm.xTitle = "Top Mass (GeV)"
    thm.yTitle = "Number of Tops / " + str((_max - _min)/_bins) + " GeV"
    thm.Histograms = hist
    thm.xMax = _max
    thm.xMin = _min
    thm.xStep = _step
    thm.xBins = _bins
    thm.Filename = "top-mass-pdgid-leptonic-" + constr_fx.__name__
    thm.dump()
    thm.SaveFigure()

    hist = [template(hadronic[k], r"$" + k + "$", c, None, {"ErrorBars" : False}) for k, c in zip(frh, colors())]
    thm = template(None, "Invariant Mass of Hadronically Matched Tops by Symbolic Decay at " + _title(level, fancy), None, mode)
    thm.Overflow = False
    thm.Stacked = False
    thm.ErrorBars = False
    thm.xTitle = "Top Mass (GeV)"
    thm.yTitle = "Number of Tops / " + str((_max - _min)/_bins) + " GeV"
    thm.Histograms = hist
    thm.xMax = _max
    thm.xMin = _min
    thm.xStep = _step
    thm.xBins = _bins
    thm.Filename = "top-mass-pdgid-hadronic-" + constr_fx.__name__
    thm.dump()
    thm.SaveFigure()

    hist = [template(wrong[k], r"$" + k + "$", c, None, {"ErrorBars" : False}) for k, c in zip(wrg, colors())]
    if not len(hist): return data

    thm = template(None, "Invariant Top Mass Distribution for Falsely Matched at " + _title(level, fancy), None, mode)
    thm.Overflow = False
    thm.Stacked = False
    thm.ErrorBars = False
    thm.xTitle = "Top Mass (GeV)"
    thm.yTitle = "Number of Tops / " + str((_max - _min)/_bins) + " GeV"
    thm.Histograms = hist
    thm.xMax = _max
    thm.xMin = _min
    thm.xStep = _step
    thm.xBins = _bins
    thm.Filename = "top-mass-pdgid-wrong-" + constr_fx.__name__
    thm.dump()
    thm.SaveFigure()
    return data

def constrain_top_njets(data, prefix, level, fancy, mode, plt, _min, _max, _bins, _step):
    mass = make(data, prefix, "top_mass")
    mult = make(data, prefix, "merged_top_jets")
    pdgi = make(data, prefix, "pdgid")
    njet = make(data, prefix, "num_jets")
    islp = make(data, prefix, "isleptonic")
    ishp = make(data, prefix, "ishadronic")

    njets_tops = {"all" : {}, "leptonic" : {}, "hadronic" : {}, "wrong" : {}}
    ntops_tops = {"all" : {}, "leptonic" : {}, "hadronic" : {}, "wrong" : {}}
    print("--------------- Contrain top Njets-------------------")

    for i in tqdm.trange(len(mass)):
        for j in range(len(mass[i])):
            try: kx = njets_format(njet[i][j])
            except IndexError: continue

            tx = ntops_format(mult[i][j])
            mx, lx, hx = mass[i][j], islp[i][j], ishp[i][j]

            key = "leptonic" if lx else "hadronic"
            if lx == hx: key = "wrong"

            if kx not in njets_tops["all"]: njets_tops["all"][kx] = []
            if kx not in njets_tops[key]:   njets_tops[key][kx] = []

            if tx not in ntops_tops["all"]: ntops_tops["all"][tx] = []
            if tx not in ntops_tops[key]:   ntops_tops[key][tx] = []

            njets_tops["all"][kx].append(mx)
            njets_tops[key][kx].append(mx)

            ntops_tops["all"][tx].append(mx)
            ntops_tops[key][tx].append(mx)


    for i in list(njets_tops["all"]):
        if i not in njets_tops["leptonic"]: njets_tops["leptonic"][i] = []
        if i not in njets_tops["hadronic"]: njets_tops["hadronic"][i] = []

    for i in list(ntops_tops["all"]):
        if i not in ntops_tops["leptonic"]: ntops_tops["leptonic"][i] = []
        if i not in ntops_tops["hadronic"]: ntops_tops["hadronic"][i] = []


    prmx = {
            "Stacked" : True, "ErrorBars" : False, "Overflow" : True, "xTitle" : "Top Mass (GeV)", 
            "yTitle" : "Number of Tops / " + str((_max - _min)/_bins) + " GeV", 
            "xMax" : _max, "xMin" : _min, "xStep": _step, "xBins" : _bins
    }

    njets_udr, njets_ovr = get_overflow(njets_tops, _max, _min)
    ntops_udr, ntops_ovr = get_overflow(ntops_tops, _max, _min)

    out = {
            "tops-merged" : {"over" : countv(ntops_ovr), "under": countv(ntops_udr), "domain" : countv(ntops_tops)}, 
            "tops-njets"  : {"over" : countv(njets_ovr), "under": countv(njets_udr), "domain" : countv(njets_tops)},
    }
    if not plt: return out

    coli = colors()
    hist = [template(njets_tops["all"][k], r"$" + k + "$", next(coli), None, {"ErrorBars" : False}) for k in sorted(njets_tops["all"])]
    thm = template(None, "Invariant Mass of Matched Tops by Jet Multiplicity at " + _title(level, fancy), None, mode, prmx)
    thm.Stacked = False 
    thm.Histograms = hist
    thm.Filename = "top-mass-njets-all"
    thm.dump()
    thm.SaveFigure()

    coli = colors()
    hist = [template(njets_tops["leptonic"][k], r"$" + k + "$", next(coli), None, {"ErrorBars" : False}) for k in sorted(njets_tops["leptonic"])]
    thm = template(None, "Invariant Mass of Leptonically Matched Tops by Jet Multiplicity at " + _title(level, fancy), None, mode, prmx)
    thm.Histograms = hist
    thm.Stacked = False
    thm.Filename = "top-mass-njets-leptonic"
    thm.dump()
    thm.SaveFigure()

    coli = colors()
    hist = [template(njets_tops["hadronic"][k], r"$" + k + "$", next(coli), None, {"ErrorBars" : False}) for k in sorted(njets_tops["hadronic"])]
    thm = template(None, "Invariant Mass of Hadronically Matched Tops by Jet Multiplicity at " + _title(level, fancy), None, mode, prmx)
    thm.Histograms = hist
    thm.Stacked = False
    thm.Filename = "top-mass-njets-hadronic"
    thm.dump()
    thm.SaveFigure()

    coli = colors()
    hist = [template(ntops_tops["all"][k], r"$" + k + "$", next(coli), None, {"ErrorBars" : False}) for k in sorted(ntops_tops["all"])]
    thm = template(None, "Invariant Mass of Matched Tops Partitioned into n-Top Contributions at " + _title(level, fancy), None, mode, prmx)
    thm.Histograms = hist
    thm.Stacked = False
    thm.Filename = "top-mass-ntops-all"
    thm.dump()
    thm.SaveFigure()

    coli = colors()
    hist = [template(ntops_tops["leptonic"][k], r"$" + k + "$", next(coli), None, {"ErrorBars" : False}) for k in sorted(ntops_tops["leptonic"])]
    thm = template(None, "Invariant Mass of Leptonically Matched Tops Partitioned into n-Top Contributions at " + _title(level, fancy), None, mode, prmx)
    thm.Histograms = hist
    thm.Stacked = False
    thm.Filename = "top-mass-ntops-leptonic"
    thm.dump()
    thm.SaveFigure()

    coli = colors()
    hist = [template(ntops_tops["hadronic"][k], r"$" + k + "$", next(coli), None, {"ErrorBars" : False}) for k in sorted(ntops_tops["hadronic"])]
    thm = template(None, "Invariant Mass of Hadronically Matched Tops Partitioned into n-Top Contributions at " + _title(level, fancy), None, mode, prmx)
    thm.Histograms = hist
    thm.Stacked = False
    thm.Filename = "top-mass-ntops-hadronic"
    thm.dump()
    thm.SaveFigure()
    return out


def ReadData(path, tree, strings):
    iox = IO(path)
    iox.Trees = tree
    iox.Leaves = strings
    data_out = {i : [] for i in strings}
    mapping = None

    for i in iox:
        if mapping is None: mapping = {k : s for k in i for s in strings if not isinstance(k, str) and s == k.decode("utf-8").split(".")[-1]}
        for s in i: 
            try: data_out[mapping[s]].append(i[s])
            except KeyError: pass
    return data_out


def routing(mode):
    if mode == "top-partons" : return ["p_ntops", "p_ltops", "p_htops", "p_top_mass"]
    if mode == "top-children": return ["c_ntops", "c_ltops", "c_htops", "c_top_mass", "c_isleptonic", "c_ishadronic", "c_pdgid"]
    if mode == "top-truthjet": return ["tj_ntops", "tj_ltops", "tj_htops", "tj_top_mass", "tj_isleptonic", "tj_ishadronic", "tj_pdgid", "tj_num_jets", "tj_merged_top_jets", "tj_num_false", "tj_wrong_match"]
    if mode == "top-jets-children": return ["jc_ntops", "jc_ltops", "jc_htops", "jc_top_mass", "jc_isleptonic", "jc_ishadronic", "jc_pdgid", "jc_num_jets", "jc_merged_top_jets", "jc_num_false", "jc_wrong_match"]
    if mode == "top-jets-leptons":  return ["jl_ntops", "jl_ltops", "jl_htops", "jl_top_mass", "jl_isleptonic", "jl_ishadronic", "jl_pdgid", "jl_num_jets", "jl_merged_top_jets", "jl_num_false", "jl_wrong_match"]
    return []

def fetch_data(data, level): return {i : data[i] for i in routing(level)}

def fetch_all(pth, tree, level):
    string = []
    for i in level: string += routing(i)
    return ReadData(pth, tree, string)




