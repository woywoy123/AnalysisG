from AnalysisG.core.plotting import TH1F
global figure_path
global mass_point

def path(hist):
    hist.OutputDirectory = figure_path + "/decaymodes/" + mass_point
    return hist

def resonance_decay_modes(data):
    th_hh = TH1F()
    th_hh.Title = "Hadronic"
    th_hh.xData = data["HH"]

    th_hl = TH1F()
    th_hl.Title = "Hadronic-Leptonic"
    th_hl.xData = data["HL"]

    th_ll = TH1F()
    th_ll.Title = "Leptonic"
    th_ll.xData = data["LL"]

    th = path(TH1F())
    th.Title = "Normalized Invariant Mass Distribution of Resonance \n using Truth Children Segmented by Decay Mode"
    th.Histograms = [th_hh, th_hl, th_ll]
    th.xMin = 0
    th.xMax = 1200
    th.xBins = 300
    th.xStep = 100
    th.Stacked = True
    th.Density = True
    th.yTitle = "Density (Arb.) / ($4$ GeV)"
    th.xTitle = "Invariant Mass (GeV)"
    th.Filename = "Figure.4.a"
    th.SaveFigure()

def pdgid_modes(ana):
    rdata = ana.res_top_pdgid
    sdata = ana.spec_top_pdgid
    alldata = ana.all_pdgid
    ntops = float(sum(ana.ntops))

    all_keys = list(alldata)
    for a in all_keys:
        if a not in rdata: rdata[a] = 0
        if a not in sdata: sdata[a] = 0
    rdata = {a : rdata[a] / ntops for a in all_keys}
    sdata = {a : sdata[a] / ntops for a in all_keys}

    rth = TH1F()
    rth.Title = "Resonance-Tops"
    rth.xLabels = rdata

    sth = TH1F()
    sth.Title = "Spectator-Tops"
    sth.xLabels = sdata

    th = path(TH1F())
    th.Histograms = [rth, sth]
    th.xLabels = {i : 0 for a, i in enumerate(alldata)}
    th.Title = "PDGID Codes of Tops Decaying Truth Partons (Children)"
    th.xTitle = "Symbol"
    th.yTitle = "Fraction"
    th.Filename = "Figure.4.b"
    th.Stacked = True
    th.SaveFigure()

    frac = {a : rdata[a] + sdata[a] for a in all_keys}
    frac = {a : k*100 for a, k in frac.items()}

def regions(ana):
    ss = ana["SS"]
    so = ana["SO"]

    rth = TH1F()
    rth.Title = "Same Sign"
    rth.xData = ss

    sth = TH1F()
    sth.Title = "Opposite Sign"
    sth.xData = so

    th = path(TH1F())
    th.Histograms = [rth, sth]
    th.xBins = 500
    th.xMin = 0
    th.xMax = 1500
    th.xStep = 150
    th.Title = "Invariant Mass Distribution of the Resonance when \n one Spectator and Resonant Top Decay Leptonically"
    th.xTitle = "Invariant Mass (GeV)"
    th.yTitle = "Entries (Arb.)"
    th.Filename = "Figure.4.c"
    th.SaveFigure()


def region_stats(ana):
    total = sum(ana.lepton_statistics.values())
    Zprod = {"Z -> HH": 0, "Z -> LH" : 0, "Z -> LL" : 0}
    tttt = {"0L" : 0, "1L" : 0, "2L" : 0, "2L (OS)" : 0, "2L (SS)" : 0, "3L" : 0, "4L" : 0}
    for i in ana.lepton_statistics:
        if i.count("l") == 0: tttt["0L"] += ana.lepton_statistics[i]
        if i.count("l") == 1: tttt["1L"] += ana.lepton_statistics[i]
        if i.count("l") == 2: tttt["2L"] += ana.lepton_statistics[i]
        if i.count("l") == 3: tttt["3L"] += ana.lepton_statistics[i]
        if i.count("l") == 4: tttt["4L"] += ana.lepton_statistics[i]

        if i.count("l") == 2:
            if i.count("(+)") == 2 or i.count("(-)") == 2: tttt["2L (SS)"] += ana.lepton_statistics[i]
            if i.count("(-)") == 1 and i.count("(+)") == 1: tttt["2L (OS)"] += ana.lepton_statistics[i]
            if "l(+)R" in i and "l(+)S" in i: Zprod["Z -> LH"] += ana.lepton_statistics[i]
            if "l(-)R" in i and "l(-)S" in i: Zprod["Z -> LH"] += ana.lepton_statistics[i]
            if "l(+)R" in i and "l(-)R" in i: Zprod["Z -> LL"] += ana.lepton_statistics[i]
            if i.count("hR") == 2: Zprod["Z -> HH"] += ana.lepton_statistics[i]
    print(total)
    print({i : (Zprod[i] / total)*100 for i in Zprod})
    print({i : (tttt[i] / total)*100 for i in tttt})
    exit()






def DecayModes(ana):
    resonance_decay_modes(ana.res_top_modes)
    pdgid_modes(ana)
    regions(ana.signal_region)
    region_stats(ana)
