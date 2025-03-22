from AnalysisG.core.plotting import TH1F, TH2F
from AnalysisG.core.io import IO

def mrg(p, m = 7): return "".join([" "]*(m - len(str(p)))) + str(p)
def colors(cl = ["red", "green", "orange", "purple", "pink"]): return iter(cl)
def count(data): return {c : sum([c == t for t in data]) for c in list(set(data))}
def countv(data):  return {c : {j : len(data[c][j]) for j in data[c]} for c in data}
def toGeV(data): return [x / 1000.0 for x in data]
def loss(t,  f, dm = None): 
    lx = ((t - f) / t)*100.0
    if lx < 0: lx = 100 + abs(lx)
    if dm is None: return lx
    return mrg(round(lx, dm), dm + 4)

def pc(t, f, m = None): return mrg(100*f / t) if m is None else mrg(100*round(f / t, m), m)

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
    
def get_overflow(data, max_, min_):
    overflw = {i : {j : [] for j in data[i]} for i in data}
    undrflw = {i : {j : [] for j in data[i]} for i in data}
    for key in data:
        for kx in data[key]: 
            overflw[key][kx] = [v for v in data[key][kx] if v >= max_]
            undrflw[key][kx] = [v for v in data[key][kx] if v <= min_]
            data[key][kx]    = [v for v in data[key][kx] if v > min_ and v < max_]
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
    thl.ErrorBars = True
    thl.Title = title
    for i in params: setattr(thl, i, params[i])
    return thl

def top_decay_stats(data, ntop, ltop, htop, level, fancy, mode, plt):
    num_tops , num_ltop ,  num_htop = data[ntop]   , data[ltop]   , data[htop]
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
    tpl.SaveFigure()
    return {"ntops" : init_tops, "leptonic" : init_ltop, "hadronic": init_htop, "counts": count(num_tops)}

def top_mass_dist(data, prefix, level, fancy, mode, plt, _min, _max, _bins, _step):
    ml_top = make(data, prefix if not "p" else None, "isleptonic")
    mh_top = make(data, prefix if not "p" else None, "ishadronic")
    m_top  = make(data, prefix, "top_mass")

    if not plt: return 
    ml_top = sum(ml_top, []) if ml_top is not None else []
    mh_top = sum(mh_top, []) if mh_top is not None else []
    m_top  = sum(m_top, [])

    _mode = {
            "Hadronic" : [m for m, c in zip(m_top, mh_top) if c], 
            "Leptonic" : [m for m, c in zip(m_top, ml_top) if c], 
            "Debug"    : [m for m, l, h in zip(m_top, ml_top, mh_top) if (l+h) == 0]
    }
    if len(_mode["Debug"]): print("DEBUG!!!"); exit()
    
    hist = []
    if len(mh_top): hist.append(template(_mode["Leptonic"], "Leptonic", "red" , None))
    if len(ml_top): hist.append(template(_mode["Hadronic"], "Hadronic", "blue", None, {"Hatch" : "//////\\\\\\\\\\\\\\"}))

    thm = template(None, "Invariant Top Mass Distribution at " + _title(level, fancy), "blue", mode)
    if len(hist): thm.Histograms = hist
    if not len(hist): thm.xData = m_top
    thm.Overflow = True
    thm.xTitle = "Top Mass (GeV)"
    thm.yTitle = "Number of Tops / " + str((_max - _min)/_bins) + " GeV"
    thm.xMax = _max
    thm.xMin = _min
    thm.xStep = _step
    thm.xBins = _bins
    thm.Filename = "top-mass"
    thm.SaveFigure()

def constrain_top_mass(data, prefix, level, fancy, mode, plt, _min, _max, _bins, _step, constr_fx):
    pdgid, top_m = make(data, prefix,      "pdgid"), make(data, prefix,   "top_mass")
    is_l, is_h   = make(data, prefix, "isleptonic"), make(data, prefix, "ishadronic")

    leptonic = {}
    hadronic = {}
    for x in range(len(pdgid)):
        for y in range(len(pdgid[x])):
            sym = symbolic(pdgid[x][y])
            is_l_, is_h_ = is_l[x][y], is_h[x][y]
            if not constr_fx(sym, top_m[x][y], pdgid[x][y]): continue
            if is_l_: modes = leptonic 
            else: modes = hadronic
            if sym not in modes: modes[sym] = []
            modes[sym].append(top_m[x][y])

    ltop = len(sum([leptonic[k] for k in leptonic], []))
    htop = len(sum([hadronic[k] for k in hadronic], []))
    data = {"leptonic" : ltop, "hadronic" : htop, "ntops": ltop + htop}
    if not plt: return data


    hist = [template(leptonic[k], r"$" + k + "$", None, None, {"ErrorBars" : False}) for k in leptonic]
    thm = template(None, "Invariant Top Mass (Leptonic) Distribution Partitioned into Symbolic Decay at " + _title(level, fancy), None, mode)
    thm.Overflow = True
    thm.Stacked = True
    thm.ErrorBars = False
    thm.xTitle = "Top Mass (GeV)"
    thm.yTitle = "Number of Tops / " + str((_max - _min)/_bins) + " GeV"
    thm.Histograms = hist
    thm.xMax = _max
    thm.xMin = _min
    thm.xStep = _step
    thm.xBins = _bins
    thm.Filename = "top-mass-pdgid-leptonic-" + constr_fx.__name__
    thm.SaveFigure()

    hist = [template(hadronic[k], r"$" + k + "$", None, None, {"ErrorBars" : False}) for k in hadronic]
    thm = template(None, "Invariant Top Mass (Hadronic) Distribution Partitioned into Symbolic Decay at " + _title(level, fancy), None, mode)
    thm.Overflow = True
    thm.Stacked = True
    thm.ErrorBars = False
    thm.xTitle = "Top Mass (GeV)"
    thm.yTitle = "Number of Tops / " + str((_max - _min)/_bins) + " GeV"
    thm.Histograms = hist
    thm.xMax = _max
    thm.xMin = _min
    thm.xStep = _step
    thm.xBins = _bins
    thm.Filename = "top-mass-pdgid-hadronic-" + constr_fx.__name__
    thm.SaveFigure()
    return data

def constrain_top_njets(data, prefix, level, fancy, mode, plt, _min, _max, _bins, _step):
    mass = make(data, prefix, "top_mass")
    mult = make(data, prefix, "merged_top_jets")
    pdgi = make(data, prefix, "pdgid")
    njet = make(data, prefix, "num_jets")
    islp = make(data, prefix, "isleptonic")

    njets_tops = {"all" : {}, "leptonic" : {}, "hadronic" : {}}
    ntops_tops = {"all" : {}, "leptonic" : {}, "hadronic" : {}}

    for i in range(len(mass)):
        for j in range(len(mass[i])):
            kx = njets_format(njet[i][j])
            tx = ntops_format(mult[i][j])
            mx, lx = mass[i][j], islp[i][j]

            key = "leptonic" if lx else "hadronic"
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
    thm = template(None, "Invariant Top Mass Partitioned into Jet Multiplicity at " + _title(level, fancy), None, mode, prmx)
    thm.Histograms = hist
    thm.Filename = "top-mass-njets-all"
    thm.SaveFigure()

    coli = colors()
    hist = [template(njets_tops["leptonic"][k], r"$" + k + "$", next(coli), None, {"ErrorBars" : False}) for k in sorted(njets_tops["leptonic"])]
    thm = template(None, "Invariant Top Mass Partitioned into Jet Multiplicity (Leptonic) at " + _title(level, fancy), None, mode, prmx)
    thm.Histograms = hist
    thm.Filename = "top-mass-njets-leptonic"
    thm.SaveFigure()

    coli = colors()
    hist = [template(njets_tops["hadronic"][k], r"$" + k + "$", next(coli), None, {"ErrorBars" : False}) for k in sorted(njets_tops["hadronic"])]
    thm = template(None, "Invariant Top Mass Partitioned into Jet Multiplicity (Hadronic) at " + _title(level, fancy), None, mode, prmx)
    thm.Histograms = hist
    thm.Filename = "top-mass-njets-hadronic"
    thm.SaveFigure()

    coli = colors()
    hist = [template(ntops_tops["all"][k], r"$" + k + "$", next(coli), None, {"ErrorBars" : False}) for k in sorted(ntops_tops["all"])]
    thm = template(None, "Invariant Top Mass Partitioned into Merged Top Multiplicity at " + _title(level, fancy), None, mode, prmx)
    thm.Histograms = hist
    thm.Filename = "top-mass-ntops-all"
    thm.SaveFigure()

    coli = colors()
    hist = [template(ntops_tops["leptonic"][k], r"$" + k + "$", next(coli), None, {"ErrorBars" : False}) for k in sorted(ntops_tops["leptonic"])]
    thm = template(None, "Invariant Top Mass Partitioned into Merged Top Multiplicity (Leptonic) at " + _title(level, fancy), None, mode, prmx)
    thm.Histograms = hist
    thm.Filename = "top-mass-ntops-leptonic"
    thm.SaveFigure()

    coli = colors()
    hist = [template(ntops_tops["hadronic"][k], r"$" + k + "$", next(coli), None, {"ErrorBars" : False}) for k in sorted(ntops_tops["hadronic"])]
    thm = template(None, "Invariant Top Mass Partitioned into Merged Top Multiplicity (Hadronic) at " + _title(level, fancy), None, mode, prmx)
    thm.Histograms = hist
    thm.Filename = "top-mass-ntops-hadronic"
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
    if mode == "top-truthjet": return ["tj_ntops", "tj_ltops", "tj_htops", "tj_top_mass", "tj_isleptonic", "tj_ishadronic", "tj_pdgid", "tj_num_jets", "tj_merged_top_jets"]
    return []

def fetch_data(data, level): return {i : data[i] for i in routing(level)}

def fetch_all(pth, tree, level):
    string = []
    for i in level: string += routing(i)
    return ReadData(pth, tree, string)




