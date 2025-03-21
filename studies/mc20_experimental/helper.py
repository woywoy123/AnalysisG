from AnalysisG.core.plotting import TH1F, TH2F
from AnalysisG.core.io import IO

def count(data): return {c : sum([c == t for t in data]) for c in list(set(data))}
def toGeV(data): return [x / 1000.0 for x in data]
def loss(t,  f): return ((t - f) / t)*100.0
def _title(level, fancy): return level + "\n (" + fancy + ")"
def safe(data, key):
    try: return data[key]
    except: return None

def bar(p, sym): return sym if p > 0 else "\\bar{" + sym + "}"
def sorted_index(seq): return [i[0] for i in sorted(enumerate(seq), key=lambda x:x[1])]

def symbolic(p):
    if isinstance(p, list): return ", ".join([symbolic(p[k]) for k in sorted_index([abs(i) for i in p])])
    if abs(p) == 1:  return bar(p, "d") #"\\texttt{light}" #
    if abs(p) == 2:  return bar(p, "u") #"\\texttt{light}" #
    if abs(p) == 3:  return bar(p, "s") #"\\texttt{light}" #
    if abs(p) == 4:  return bar(p, "c") #"\\texttt{light}" #
    if abs(p) == 5:  return bar(p, "b")
    if abs(p) == 11: return bar(p, "e")
    if abs(p) == 12: return bar(p, "\\nu_{e}")
    if abs(p) == 13: return bar(p, "\\mu")
    if abs(p) == 14: return bar(p, "\\nu_{\\mu}")
    if abs(p) == 15: return bar(p, "\\tau")
    if abs(p) == 16: return bar(p, "\\nu_{\\tau}")
    if abs(p) == 21: return bar(p, "g")
    if abs(p) == 22: return bar(p, "\\gamma")
    return "undef"

def enforce_sym(p):
    if "gamma" in p: return False
    return True

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
    if not plt: return init_tops, init_ltop, init_htop, count(num_tops)

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
    return init_tops, init_ltop, init_htop, count(num_tops)

def top_mass_dist(data, ntop, ltop, htop, level, fancy, mode, plt, _min, _max, _bins, _step):
    m_top, ml_top, mh_top = safe(data, ntop), safe(data, ltop), safe(data, htop)
    if not plt: return 
    ml_top = sum(ml_top, []) if ml_top is not None else []
    mh_top = sum(mh_top, []) if mh_top is not None else []
    m_top  = sum(m_top, [])

    _mode = {
            "Hadronic" : [m for m, c in zip(m_top, mh_top) if c], 
            "Leptonic" : [m for m, c in zip(m_top, ml_top) if c], 
            "Debug"    : [m for m, l, h in zip(m_top, ml_top, mh_top) if (l+h) == 0]
    }
    if len(_mode["Debug"]): print("DEBUG!!!")
    print(_mode["Leptonic"])
    
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

def constrain_top_mass(data, pdgid, top_m, level, fancy, mode, plt, _min, _max, _bins, _step):
    pdgid, top_m = safe(data, pdgid), safe(data, top_m)
   
    modes = {}
    for x in range(len(pdgid)):
        for y in range(len(pdgid[x])):
            sym = symbolic(pdgid[x][y])
            if not enforce_sym(sym): continue
            if sym not in modes: modes[sym] = []
            modes[sym].append(top_m[x][y])

    hist = [template(modes[k], r"$" + k + "$", None, None, {"ErrorBars" : False}) for k in modes]
    thm = template(None, "Invariant Top Mass Distribution Partitioned into Symbolic Decay at " + _title(level, fancy), None, mode)
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
    thm.Filename = "top-mass-pdgid"
    thm.SaveFigure()

    
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

def fetch_data(pth, tree, level):
    if level == "top-partons" : string = ["p_ntops", "p_ltops", "p_htops", "p_top_mass"]
    if level == "top-children": string = ["c_ntops", "c_ltops", "c_htops", "c_top_mass", "c_isleptonic", "c_ishadronic", "c_pdgid"]
    if level == "top-truthjet": string = ["tj_ntops", "tj_ltops", "tj_htops", "tj_top_mass", "tj_isleptonic", "tj_ishadronic", "tj_pdgid", "tj_num_jets", "tj_merged_top_jets"]
    return ReadData(pth, tree, string)

