from AnalysisG.core.plotting import TH1F, TH2F
colors = ["red", "green", "blue", "orange", "magenta", "cyan", "pink"]

def path(hist):
    hist.Style = r"ATLAS"
    hist.OutputDirectory = "figures"
    hist.DPI = 250
    hist.TitleSize = 20
    hist.AutoScaling = True
    hist.Overflow = False
    hist.yScaling = 10
    hist.xScaling = 15
    hist.FontSize = 15
    hist.AxisSize = 14
    return hist

def template_mass(data):
    thl = TH1F()
    thl.xData = data["leptonic"]
    thl.Title = "Leptonic"
    thl.Color = "blue"
    thl.Alpha = 0.5
    thl.Density = True
    thl.ErrorBars = True

    thh = TH1F()
    thh.xData = data["hadronic"]
    thh.Title = "Hadronic"
    thh.Color = "orange"
    thh.Hatch = "//"
    thh.Alpha = 0.5
    thh.Density = True
    thh.ErrorBars = True

    thd = path(TH1F())
    thd.UseLateX = True
    thd.yTitle = "Normalized to Number of Tops / 1 GeV"
    thd.xTitle = "Invariant Mass of Top (GeV)"
    thd.Histograms = [thl, thh]
    thd.ErrorBars = True
    return thd

def template_delta(data, typex, formal):
    thl = path(TH2F())
    thl.Title = "Error of Top Matching Scheme for \n " + formal
    thl.xMin = 0
    thl.xMax = 1000
    thl.xStep = 100
    thl.xBins = 100
    thl.xTitle = r"Error of Top Matched Four Vector ($\chi$ GeV)"
    thl.xData = data["chi2"][typex]

    thl.yMin = 0
    thl.yMax = 100
    thl.yStep = 10
    thl.yBins = 100
    thl.yTitle = r"Fractional Invariant Mass Error of Derived Top $(|(\texttt{T}^{\texttt{M}}_{\texttt{Drv}} - \texttt{T}^{\texttt{M}}_{\texttt{Tru}}) / \texttt{T}^{\texttt{M}}_{\texttt{Tru}}|)$ (Arb.)"
    thl.yData = [abs(i) for i in data["delta-mass"][typex]]
    return thl



def chi(tp, tc):
    pairs = []
    for i in tc:
        ht = i.top_hash
        dct = {}
        for j in tp:
            if j.top_hash != ht: continue
            x  = ((i.px - j.px) / 1000.0)**2
            x += ((i.py - j.py) / 1000.0)**2
            x += ((i.pz - j.pz) / 1000.0)**2
            x += ((i.e  - j.e ) / 1000.0)**2
            dct[x**0.5] = [i, j]
        if not len(dct): continue
        sx = sorted(dct)[0]
        _tp, tt = dct[sx]
        delM = (_tp.Mass - tt.Mass)/tt.Mass
        pairs.append([sx, delM] + dct[sx])
    return pairs

def top_decay_stats(sel, mode, formal, plt):
    hists = []

    data = {"leptonic" : [], "hadronic" : [], "nleptonic" : [], "nhadronic" : [], "ntops" : []}  
    for i in sel.truth_tops:
        nlep = 0
        nhad = 0
        ntop = 0
        for t in i:
            mass = t.Mass / 1000.0
            is_lep = sum([c.is_lep for c in t.Children])
            data["leptonic" if is_lep else "hadronic"] += [mass]
            nlep += is_lep > 0
            nhad += is_lep == 0
            ntop += 1
        data["nleptonic"] += [nlep]
        data["nhadronic"] += [nhad]
        data["ntops"]     += [ntop]
   
    num_tops = sum(data["ntops"])
    nleps = sum(data["nleptonic"])
    nhadr = sum(data["nhadronic"])

    print("------ Initial Top Parton Level --------")
    print("Hadronic Decay Fraction: " + str(nhadr/num_tops))
    print("Leptonic Decay Fraction: " + str(nleps/num_tops))
    print("Number of Tops: " + str(num_tops))
    if not plt: return  {"num_tops" : num_tops, "nleps" : nleps, "nhadr" : nhadr} 

    thl = TH1F()
    thl.xData = data["nleptonic"]
    thl.Title = "Leptonic"
    thl.Color = "blue"
    thl.Alpha = 0.5
    thl.ErrorBars = True

    thh = TH1F()
    thh.xData = data["nhadronic"]
    thh.Title = "Hadronic"
    thh.Color = "orange"
    thh.Hatch = "//"
    thh.Alpha = 0.5
    thh.ErrorBars = True

    thd = path(TH1F())
    thd.UseLateX = True
    thd.Title = "Number of Tops per Event Decaying Hadronically or Leptonically at Parton Level"
    thd.xMin = 0
    thd.xMax = 5
    thd.xBins = 5
    thd.xStep = 1
    thd.yTitle = "Events (Arb.)"
    thd.xTitle = "Number of Tops / Event"
    thd.Histograms = [thl, thh]
    thd.Filename = "top-decay-" + mode
    thd.ErrorBars = True
    thd.SaveFigure()

    thd = template_mass(data)
    thd.Title = "Invariant Mass of Top-Quark at Parton Level (" + formal + ")"
    thd.Filename = "top-mass-" + mode
    thd.xMin = 140
    thd.xMax = 200
    thd.xBins = 60
    thd.xStep = 4
    thd.SaveFigure()
    return  {"num_tops" : num_tops, "nleps" : nleps, "nhadr" : nhadr} 

def top_decay_children(sel, initial, mode, formal, plt):
    data = {
            "ntops" : [], 
            "leptonic"   : [], "hadronic"  : [], 
            "nleptonic"  : [], "nhadronic" : [], 
            "chi2"       : {"leptonic" : [], "hadronic" : []}, 
            "delta-mass" : {"leptonic" : [], "hadronic" : []}
    } 

    tps = sel.top_children
    rlt = sel.truth_tops
    for i in range(len(tps)):
        nlep = 0
        nhad = 0
        ntop = 0
        pairs = chi(rlt[i], tps[i])
        for pr in pairs:
            _chi, delM, mtc_tp, tru_tp = pr
            is_lep = sum([c.is_lep for c in tru_tp.Children])
            key = "leptonic" if is_lep else "hadronic"
            data[key] += [mtc_tp.Mass / 1000.0]
            data["chi2"][key] += [_chi]
            data["delta-mass"][key] += [delM]
            nlep += is_lep > 0
            nhad += is_lep == 0
            ntop += 1

        data["nleptonic"] += [nlep]
        data["nhadronic"] += [nhad]
        data["ntops"]     += [ntop]

    num_tops = sum(data["ntops"])
    print("------ Fraction of Matched Tops @ TopChildren Level --------")
    print("Hadronic Decay: " + str(sum(data["nhadronic"])/num_tops))
    print("Leptonic Decay: " + str(sum(data["nleptonic"])/num_tops))
    print("Number of Tops: " + str(num_tops))

    print("=========== Loss ==========")
    print("Loss Hadronic (%): " + str(abs(sum(data["nhadronic"]) - initial["nhadr"])*100/initial["nhadr"]))
    print("Loss Leptonic (%): " + str(abs(sum(data["nleptonic"]) - initial["nleps"])*100/initial["nleps"]))
    print("Loss Tops (%):     " + str(abs(num_tops - initial["num_tops"])*100 / initial["num_tops"]))

    if not plt: return
    thd = template_mass(data)
    thd.Title = "Invariant Mass of Top-Quark at TopChildren Level (" + formal + ")"
    thd.Filename = "topchildren-mass-" + mode
    thd.xMin = 0
    thd.xMax = 300
    thd.xBins = 300
    thd.xStep = 20
    thd.SaveFigure()

    thx = template_delta(data, "leptonic", "Leptonically Decaying Tops (" + formal + ")")
    thx.yMax = 5
    thx.yStep = 0.5
    thx.Filename = "error_mass_chi_lepton-topchildren-" + mode
    thx.SaveFigure()

    thx = template_delta(data, "hadronic", "Hadronically Decaying Tops (" + formal + ")")
    thx.yMax = 5
    thx.yStep = 0.5
    thx.Filename = "error_mass_chi_hadronic-topchildren-" + mode
    thx.SaveFigure()


def top_decay_truthjets(sel, initial, mode, formal, plt):
    data = {
            "ntops" : [], 
            "leptonic"   : [], "hadronic"  : [], 
            "nleptonic"  : [], "nhadronic" : [], 
            "chi2"       : {"leptonic" : [], "hadronic" : []}, 
            "delta-mass" : {"leptonic" : [], "hadronic" : []}
    } 

    tps = sel.truth_jets
    rlt = sel.truth_tops
    for i in range(len(tps)):
        nlep = 0
        nhad = 0
        ntop = 0

        pairs = chi(rlt[i], tps[i])
        for pr in pairs:
            _chi, delM, mtc_tp, tru_tp = pr
            is_lep = sum([c.is_lep for c in tru_tp.Children])
            key = "leptonic" if is_lep else "hadronic"
            data[key] += [mtc_tp.Mass / 1000.0]
            data["chi2"][key] += [_chi]
            data["delta-mass"][key] += [delM]
            nlep += is_lep > 0
            nhad += is_lep == 0
            ntop += 1

        data["nleptonic"] += [nlep]
        data["nhadronic"] += [nhad]
        data["ntops"]     += [ntop]

    num_tops = sum(data["ntops"])
    print("------ Fraction of Matched Tops @ Truth Jets Level --------")
    print("Hadronic Decay: " + str(sum(data["nhadronic"])/num_tops))
    print("Leptonic Decay: " + str(sum(data["nleptonic"])/num_tops))
    print("Number of Tops: " + str(num_tops))

    print("=========== Loss ==========")
    print("Loss Hadronic (%): " + str(abs(sum(data["nhadronic"]) - initial["nhadr"])*100/initial["nhadr"]))
    print("Loss Leptonic (%): " + str(abs(sum(data["nleptonic"]) - initial["nleps"])*100/initial["nleps"]))
    print("Loss Tops (%):     " + str(abs(num_tops - initial["num_tops"])*100 / initial["num_tops"]))

    if not plt: return 
    thd = template_mass(data)
    thd.Title = "Invariant Mass of Top-Quark at Truth Jet Level (" + formal + ")"
    thd.Filename = "truthjet-mass-" + mode
    thd.xMin = 0
    thd.xMax = 300
    thd.xBins = 300
    thd.xStep = 20
    thd.SaveFigure()

    thx = template_delta(data, "leptonic", "Leptonically Decaying Tops (" + formal + ")")
    thx.yMax = 5
    thx.yStep = 0.5
    thx.Filename = "error_mass_chi_lepton-truthjets-" + mode
    thx.SaveFigure()

    thx = template_delta(data, "hadronic", "Hadronically Decaying Tops (" + formal + ")")
    thx.yMax = 5
    thx.yStep = 0.5
    thx.Filename = "error_mass_chi_hadronic-truthjets-" + mode
    thx.SaveFigure()

def top_decay_jetschildren(sel, initial, mode, formal, plt):
    data = {
            "ntops" : [], 
            "leptonic"   : [], "hadronic"  : [], 
            "nleptonic"  : [], "nhadronic" : [], 
            "chi2"       : {"leptonic" : [], "hadronic" : []}, 
            "delta-mass" : {"leptonic" : [], "hadronic" : []}
    } 

    tps = sel.jets_children
    rlt = sel.truth_tops
    for i in range(len(tps)):
        nlep = 0
        nhad = 0
        ntop = 0

        pairs = chi(rlt[i], tps[i])
        for pr in pairs:
            _chi, delM, mtc_tp, tru_tp = pr
            is_lep = sum([c.is_lep for c in tru_tp.Children])
            key = "leptonic" if is_lep else "hadronic"
            data[key] += [mtc_tp.Mass / 1000.0]
            data["chi2"][key] += [_chi]
            data["delta-mass"][key] += [delM]
            nlep += is_lep > 0
            nhad += is_lep == 0
            ntop += 1

        data["nleptonic"] += [nlep]
        data["nhadronic"] += [nhad]
        data["ntops"]     += [ntop]

    num_tops = sum(data["ntops"])
    print("------ Fraction of Matched Tops @ Jets Children Level --------")
    print("Hadronic Decay: " + str(sum(data["nhadronic"])/num_tops))
    print("Leptonic Decay: " + str(sum(data["nleptonic"])/num_tops))
    print("Number of Tops: " + str(num_tops))

    print("=========== Loss ==========")
    print("Loss Hadronic (%): " + str(abs(sum(data["nhadronic"]) - initial["nhadr"])*100/initial["nhadr"]))
    print("Loss Leptonic (%): " + str(abs(sum(data["nleptonic"]) - initial["nleps"])*100/initial["nleps"]))
    print("Loss Tops (%):     " + str(abs(num_tops - initial["num_tops"])*100 / initial["num_tops"]))

    if not plt: return
    thd = template_mass(data)
    thd.Title = "Invariant Mass of Top-Quark at Truth Jet Level (" + formal + ")"
    thd.Filename = "jetschildren-mass-" + mode
    thd.xMin = 0
    thd.xMax = 300
    thd.xBins = 300
    thd.xStep = 20
    thd.SaveFigure()

    thx = template_delta(data, "leptonic", "Leptonically Decaying Tops (" + formal + ")")
    thx.yMax = 5
    thx.yStep = 0.5
    thx.Filename = "error_mass_chi_lepton-jetschildren-" + mode
    thx.SaveFigure()

    thx = template_delta(data, "hadronic", "Hadronically Decaying Tops (" + formal + ")")
    thx.yMax = 5
    thx.yStep = 0.5
    thx.Filename = "error_mass_chi_hadronic-jetschildren-" + mode
    thx.SaveFigure()

def top_decay_jetsleptons(sel, initial, mode, formal, plt):
    data = {
            "ntops" : [], 
            "leptonic"   : [], "hadronic"  : [], 
            "nleptonic"  : [], "nhadronic" : [], 
            "chi2"       : {"leptonic" : [], "hadronic" : []}, 
            "delta-mass" : {"leptonic" : [], "hadronic" : []}
    } 

    tps = sel.jets_leptons
    rlt = sel.truth_tops
    for i in range(len(tps)):
        nlep = 0
        nhad = 0
        ntop = 0

        pairs = chi(rlt[i], tps[i])
        for pr in pairs:
            _chi, delM, mtc_tp, tru_tp = pr
            is_lep = sum([c.is_lep for c in tru_tp.Children])
            key = "leptonic" if is_lep else "hadronic"
            data[key] += [mtc_tp.Mass / 1000.0]
            data["chi2"][key] += [_chi]
            data["delta-mass"][key] += [delM]
            nlep += is_lep > 0
            nhad += is_lep == 0
            ntop += 1

        data["nleptonic"] += [nlep]
        data["nhadronic"] += [nhad]
        data["ntops"]     += [ntop]

    num_tops = sum(data["ntops"])
    print("------ Fraction of Matched Tops @ Jets Leptons Level --------")
    print("Hadronic Decay: " + str(sum(data["nhadronic"])/num_tops))
    print("Leptonic Decay: " + str(sum(data["nleptonic"])/num_tops))
    print("Number of Tops: " + str(num_tops))

    print("=========== Loss ==========")
    print("Loss Hadronic (%): " + str(abs(sum(data["nhadronic"]) - initial["nhadr"])*100/initial["nhadr"]))
    print("Loss Leptonic (%): " + str(abs(sum(data["nleptonic"]) - initial["nleps"])*100/initial["nleps"]))
    print("Loss Tops (%):     " + str(abs(num_tops - initial["num_tops"])*100 / initial["num_tops"]))

    thd = template_mass(data)
    thd.Title = "Invariant Mass of Top-Quark at Truth Jet Level (" + formal + ")"
    thd.Filename = "jetsleptons-mass-" + mode
    thd.xMin = 0
    thd.xMax = 300
    thd.xBins = 300
    thd.xStep = 20
    thd.SaveFigure()

    thx = template_delta(data, "leptonic", "Leptonically Decaying Tops (" + formal + ")")
    thx.yMax = 5
    thx.yStep = 0.5
    thx.Filename = "error_mass_chi_lepton-jetsleptons-" + mode
    thx.SaveFigure()

    thx = template_delta(data, "hadronic", "Hadronically Decaying Tops (" + formal + ")")
    thx.yMax = 5
    thx.yStep = 0.5
    thx.Filename = "error_mass_chi_hadronic-jetsleptons-" + mode
    thx.SaveFigure()

def entry_point(sel, mode, formal, plt = True):
    print("========================== (" + formal + ") =========================")
    init = top_decay_stats(sel, mode, formal, plt)
    print("")
    top_decay_children(sel, init, mode, formal, plt)
    print("")
    top_decay_truthjets(sel, init, mode, formal, plt)
    print("")
    top_decay_jetschildren(sel, init, mode, formal, plt)
    print("")
    top_decay_jetsleptons(sel, init, mode, formal, plt)
    print("===================================================================")

def entry(exp_mc20 = None, ref_mc20 = None, ref_mc16 = None):
    if exp_mc20 is not None: entry_point(exp_mc20, "exp_mc20", "Experimental MC20")
    if ref_mc20 is not None: entry_point(ref_mc20, "ref_mc20", "Top-CP MC20"      )
    if ref_mc16 is not None: entry_point(ref_mc16, "ref_mc16", "Reference MC16"   )



