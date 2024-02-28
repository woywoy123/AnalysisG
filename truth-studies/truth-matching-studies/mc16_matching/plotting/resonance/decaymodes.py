from AnalysisG.Plotting import TH1F

def settings():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : "./plt_plots/resonance/",
            "Histograms" : [],
            "Histogram" : None,
            "LegendLoc" : "upper right"
    }
    return settings

def resonance_decay_modes(data):
    sett = settings()

    th_hh = TH1F()
    th_hh.Title = "hadronic"
    th_hh.xData = data["HH"]

    th_hl = TH1F()
    th_hl.Title = "hadronic-leptonic"
    th_hl.xData = data["HL"]

    th_ll = TH1F()
    th_ll.Title = "leptonic"
    th_ll.xData = data["LL"]

    th = TH1F(**sett)
    th.Title = "Decay mode of Resonant Tops"
    th.Histograms = [th_hh, th_hl, th_ll]
    th.xMin = 0
    th.xMax = 1500
    th.xBins = 1500
    th.xStep = 250
    th.Stack = True
    th.yTitle = "Entries (arb.)"
    th.xTitle = "Invariant Mass (GeV)"
    th.Filename = "Figure.2.a"
    th.SaveFigure()

def resonance_decay_charge(data):
    sett = settings()

    th_so = TH1F()
    th_so.Title = "Opposite-Sign"
    th_so.xData = data["SO"]

    th_ss = TH1F()
    th_ss.Title = "Same-Sign"
    th_ss.xData = data["SS"]

    th = TH1F(**sett)
    th.Title = "Leptonically Decaying Resonance Tops in the Opposite and Same Sign Decay Channels"
    th.Histograms = [th_so, th_ss]
    th.xMin = 0
    th.xMax = 1500
    th.xBins = 1500
    th.xStep = 250
    th.xTitle = "Invariant Mass (GeV)"
    th.yTitle = "Entries (arb.)"
    th.Filename = "Figure.2.b"
    th.SaveFigure()

def spectator_decay_modes(data):
    sett = settings()

    th_hh = TH1F()
    th_hh.Title = "hadronic"
    th_hh.xData = data["HH"]

    th_hl = TH1F()
    th_hl.Title = "hadronic-leptonic"
    th_hl.xData = data["HL"]

    th_ll = TH1F()
    th_ll.Title = "leptonic"
    th_ll.xData = data["LL"]

    th = TH1F(**sett)
    th.Title = "Decay mode of Spectator Tops"
    th.Histograms = [th_hh, th_hl, th_ll]
    th.xMin = 0
    th.xMax = 1500
    th.xBins = 1500
    th.xStep = 250
    th.xTitle = "Invariant Mass (GeV)"
    th.yTitle = "Entries (arb.)"
    th.Filename = "Figure.2.c"
    th.SaveFigure()

def spectator_decay_charge(data):
    sett = settings()

    th_so = TH1F()
    th_so.Title = "Opposite-Sign"
    th_so.xData = data["SO"]

    th_ss = TH1F()
    th_ss.Title = "Same-Sign"
    th_ss.xData = data["SS"]

    th = TH1F(**sett)
    th.Title = "Leptonically Decaying Spectator Tops in the Opposite and Same Sign Decay Channels"
    th.Histograms = [th_so, th_ss]
    th.xMin = 0
    th.xMax = 1500
    th.xBins = 1500
    th.xStep = 250
    th.xTitle = "Invariant Mass (GeV)"
    th.Filename = "Figure.2.d"
    th.SaveFigure()

def pdgid_modes(ana):
    rdata = ana.res_top_pdgid
    sdata = ana.spec_top_pdgid
    alldata = ana.all_pdgid

    all_keys = list(alldata)
    for a in all_keys:
        if a not in rdata: rdata[a] = 0
        if a not in sdata: sdata[a] = 0
    rdata = {a : rdata[a] for a in all_keys}
    sdata = {a : sdata[a] for a in all_keys}

    rth = TH1F()
    rth.Title = "Resonance-Tops"
    rth.xLabels = rdata

    sth = TH1F()
    sth.Title = "Spectator-Tops"
    sth.xLabels = sdata

    ath = TH1F()
    ath.Title = "All-Tops"
    ath.xLabels = alldata

    sett = settings()
    th = TH1F(**sett)
    th.Histogram = ath
    th.Histograms = [rth, sth]
    th.xLabels = {i : 0 for a, i in enumerate(alldata)}
    th.Title = "PDGID Codes of Tops Decaying Truth Partons (Children)"
    th.xTitle = "Symbol"
    th.yTitle = "Fraction"
    th.Filename = "Figure.2.e"
    th.Normalize = True
    th.SaveFigure()

def regions(ana):
    ss = ana["SS"]
    so = ana["SO"]

    rth = TH1F()
    rth.Title = "Same Sign"
    rth.xData = ss

    sth = TH1F()
    sth.Title = "Opposite Sign"
    sth.xData = so

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = [rth, sth]
    th.xBins = 1500
    th.xMin = 0
    th.xMax = 1500
    th.xStep = 150
    th.Title = "Invariant Mass Distribution of the Resonance when \n one Spectator and Resonant Top Decay Leptonically"
    th.xTitle = "Invariant Mass (GeV)"
    th.yTitle = "Entries"
    th.Filename = "Figure.2.f"
    th.SaveFigure()

def DecayModes(ana):
    resonance_decay_modes(ana.res_top_modes)
    resonance_decay_charge(ana.res_top_charges)

    spectator_decay_modes(ana.spec_top_modes)
    spectator_decay_charge(ana.spec_top_charges)

    pdgid_modes(ana)
    regions(ana.signal_region)



