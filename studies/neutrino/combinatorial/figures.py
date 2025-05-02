from AnalysisG.core.plotting import TH1F, TH2F
from .helpers import compiler 
from .utils import *
import pandas

def statistics(collx):
    eff_all = collx.efficiency

    nominal = eff_all["nominal"]
    print("-------------- Nominal --------------- ")
    print(pandas.DataFrame.from_dict(nominal))
    
    print("----------- Chi 2 assisted ----------- ")
    chi2x   = eff_all["chi2-fitted"]
    print(pandas.DataFrame.from_dict(chi2x))

def default(hist, pth):
    hist.Style = "ATLAS"
    if pth is not None: hist.OutputDirectory = "figures/combinatorial/" + pth
    hist.DPI = 300
    hist.TitleSize = 15
    hist.AutoScaling = True
    hist.Overflow = False
    hist.yScaling = 5 #10*0.75
    hist.xScaling = 5 #15*0.6
    hist.FontSize = 10
    hist.AxisSize = 10
    hist.LegendSize = 10
    return hist

def template(xdata, title, color = None, pth = None, params = {}):
    thl = default(TH1F(), pth) if xdata is None else TH1F()
    if xdata is not None: thl.xData = xdata
    if color is not None: thl.Color = color
    thl.xMin = 0
    thl.Alpha = 0.5
    thl.Title = title
    for i in params: setattr(thl, i, params[i])
    return thl

def template2(xdata, ydata, title, pth = None, params = {}):
    thl = default(TH2F(), pth)
    thl.xData = xdata
    thl.yData = ydata
    thl.Title = title
    thl.Color = "inferno"
    for i in params: setattr(thl, i, params[i])
    return thl

def TopMassNominal(exp, name, para):
    cols = colors()
    top_correct = template(make(exp.correct   , get_rtop, get_mass), Fancy(name, "sets"  ), next(cols))
    top_swp_b   = template(make(exp.swapped_bs, get_rtop, get_mass), Fancy(name, "swp_b" ), next(cols))
    top_swp_l   = template(make(exp.swapped_ls, get_rtop, get_mass), Fancy(name, "swp_l" ), next(cols))
    top_swp_bl  = template(make(exp.swapped_bl, get_rtop, get_mass), Fancy(name, "swp_bl"), next(cols))
    top_fakes   = template(make(exp.fake_nus  , get_rtop, get_mass), Fancy(name, "fake"  ), next(cols))
    th1 = template(None, "Top Reconstruction (nominal) " + Fancy(name), None, name, para)
    th1.Histograms = sorted([top_correct, top_swp_b, top_swp_l, top_swp_bl, top_fakes], reverse = True, key = lambda x: sum(x.counts))
    th1.Filename = "all_nunu-nominal"
    th1.SaveFigure()

def TopMassChi2(exp, name, para):
    cols = colors()
    top_correct = template(make(exp.correct   , get_rtop, get_mass), Fancy(name, "sets"  ), next(cols))
    top_swp_b   = template(make(exp.swapped_bs, get_rtop, get_mass), Fancy(name, "swp_b" ), next(cols))
    top_swp_l   = template(make(exp.swapped_ls, get_rtop, get_mass), Fancy(name, "swp_l" ), next(cols))
    top_swp_bl  = template(make(exp.swapped_bl, get_rtop, get_mass), Fancy(name, "swp_bl"), next(cols))
    top_fakes   = template(make(exp.fake_nus  , get_rtop, get_mass), Fancy(name, "fake"  ), next(cols))
    th1 = template(None, "Top Reconstruction ($\\chi^2$ minimization) " + Fancy(name), None, name, para)
    th1.Histograms = sorted([top_correct, top_swp_b, top_swp_l, top_swp_bl, top_fakes], reverse = True, key = lambda x: sum(x.counts))
    th1.Filename = "all_nunu-chi2"
    th1.SaveFigure()


def TopMassUnswapped(exp, name, para):
    cols = colors()
    top_correct = template(make_atm(exp.correct   , fix_swapped, get_swp_mass), Fancy(name, "sets"  ), next(cols))
    top_swp_b   = template(make_atm(exp.swapped_bs, fix_swapped, get_swp_mass), Fancy(name, "swp_b" ), next(cols))
    top_swp_l   = template(make_atm(exp.swapped_ls, fix_swapped, get_swp_mass), Fancy(name, "swp_l" ), next(cols))
    top_swp_bl  = template(make_atm(exp.swapped_bl, fix_swapped, get_swp_mass), Fancy(name, "swp_bl"), next(cols))
    top_fakes   = template(make_atm(exp.fake_nus  , fix_swapped, get_swp_mass), Fancy(name, "fake"  ), next(cols))
    th1 = template(None, "Top Reconstruction (Unswapped) " + Fancy(name), None, name, para)
    th1.Histograms = sorted([top_correct, top_swp_b, top_swp_l, top_swp_bl, top_fakes], reverse = True, key = lambda x: sum(x.counts))
    th1.Filename = "all_nunu-unswapped"
    th1.SaveFigure()

def TopMassUnswappedChi2(exp, name, para):
    cols = colors()
    top_correct = template(make_atm(exp.correct   , fix_swapped, get_swp_mass), Fancy(name, "sets"  ), next(cols))
    top_swp_b   = template(make_atm(exp.swapped_bs, fix_swapped, get_swp_mass), Fancy(name, "swp_b" ), next(cols))
    top_swp_l   = template(make_atm(exp.swapped_ls, fix_swapped, get_swp_mass), Fancy(name, "swp_l" ), next(cols))
    top_swp_bl  = template(make_atm(exp.swapped_bl, fix_swapped, get_swp_mass), Fancy(name, "swp_bl"), next(cols))
    top_fakes   = template(make_atm(exp.fake_nus  , fix_swapped, get_swp_mass), Fancy(name, "fake"  ), next(cols))
    th1 = template(None, "Top Reconstruction (Unswapped - $\\chi^2$ minimization) " + Fancy(name), None, name, para)
    th1.Histograms = sorted([top_correct, top_swp_b, top_swp_l, top_swp_bl, top_fakes], reverse = True, key = lambda x: sum(x.counts))
    th1.Filename = "all_nunu-unswapped-chi2"
    th1.SaveFigure()

def NeutrinoAsymmetry(exp, name, para):
    modes = {}
    modes["sets"]   = make_atm(exp.correct   , fix_swapped, get_chi2_nus)
    modes["swp_b"]  = make_atm(exp.swapped_bs, fix_swapped, get_chi2_nus)
    modes["swp_l"]  = make_atm(exp.swapped_ls, fix_swapped, get_chi2_nus)
    modes["swp_bl"] = make_atm(exp.swapped_bl, fix_swapped, get_chi2_nus)
    modes["fake"]   = make_atm(exp.fake_nus  , fix_swapped, get_chi2_nus)

    for k in modes:
        data = modes[k]
        met, chx, els = {}, {}, {}
        for i in data:
            ch1, ch2, asym, elx, sym = i
            if   sym.count("e")   == 4: sym = sym.replace("e"  , "\\ell")
            elif sym.count("mu")  == 4: sym = sym.replace("mu" , "ell")
            elif sym.count("tau") == 4: sym = sym.replace("tau", "ell")
            try:   met[sym] += [asym]
            except: met[sym] = [asym]

            try:   chx[sym] += [ch1, ch2]
            except: chx[sym] = [ch1, ch2]

            try:   els[sym] += [elx]
            except: els[sym] = [elx]


        base = dict(para)
        base["yLogarithmic"] = False
        base["yTitle"] = "Reconstructed Dilepton Events / <units>"
        base["xTitle"] = "Neutrino Transverse Momentum Asymmetry ($\\Delta p_{T}$ GeV)"
        base["xBins"]  = 400
        base["xStep"]  = 50
        base["xMax"]   = 400
        base["yLogarithmic"] = False
        base["Stacked"] = True

        cols = colors()
        orx = sorted(met, reverse = True, key = lambda x: len(met[x]))[:5]
        met = resize(met, orx)
        for i in met: met[i] = template(met[i], i, next(cols) if i != "other" else "grey")
        th1 = template(None, "Transverse Neutrino Momentum Asymmetry \n" + Fancy(name, k), None, name, base)
        th1.Histograms = sorted(list(met.values()), reverse = False, key = lambda x: sum(x.counts))
        th1.Filename = k + "-neutrino-momentum-asymmetry"
        th1.SaveFigure()

        base = dict(para)
        base["yTitle"] = "Reconstructed Neutrinos / <units>"
        base["xTitle"] = "$\\chi^2$ Kinematic Error ($\\text{GeV}^2$)"
        base["xBins"]  = 200
        base["xStep"]  = 500
        base["xMax"]   = 4000
        base["yLogarithmic"] = False
        base["Stacked"] = True

        cols = colors()
        chx = resize(chx, orx)
        for i in chx: chx[i] = template(chx[i], i, next(cols) if i != "other" else "grey")
        th1 = template(None, "$\\chi^2$ Between Truth and Reconstructed Neutrino \n" + Fancy(name, k), None, name, base)
        th1.Histograms = sorted(list(chx.values()), reverse = False, key = lambda x: sum(x.counts))
        th1.Filename = k + "-chi2-error"
        th1.SaveFigure()

        base = {}
        base["yTitle"] = "Distance Between Elliptical Neutrinos"
        base["yBins"]  = 100
        base["yStep"]  = 4
        base["yMax"]   = 0
        base["yMin"]   = -40

        base["xTitle"] = "Neutrino Transverse Momentum Asymmetry ($\\Delta p_{T}$ GeV)"
        base["xBins"]  = 100
        base["xStep"]  = 50
        base["xMax"]   = 400

        els = resize(els, [])
        try: th2 = template2(sum([t.xData for t in met.values()], []), els["other"], "Neutrino Asymmetry and Elliptical Distance \n" + Fancy(name, k), name, base)
        except: continue
        th2.Filename = k + "-ellipse-met"
        th2.SaveFigure()

def DeltaR(exp, name, para):
    base = dict(para)
    para_ = dict(para)
    para_["yLogarithmic"] = False
    
    pr = "quark" if name == "top_children" else "jet"
    base["xTitle"] = "$\\Delta R$ Between Matched Objects ($\\ell$ and b-" + pr + ")"
    base["yTitle"] = "Density (Abr.) / <units>"
    base["xBins"] = 100
    base["xStep"] = 0.4
    base["xMin"] = 0
    base["xMax"] = 6
    base["Density"] = True
    base["yLogarithmic"] = False

    cols = colors()
    top_correct = template(make_atm(exp.correct   , get_dr_nus), Fancy(name, "sets"  ), next(cols))
    top_swp_b   = template(make_atm(exp.swapped_bs, get_dr_nus), Fancy(name, "swp_b" ), next(cols))
    top_swp_l   = template(make_atm(exp.swapped_ls, get_dr_nus), Fancy(name, "swp_l" ), next(cols))
    top_swp_bl  = template(make_atm(exp.swapped_bl, get_dr_nus), Fancy(name, "swp_bl"), next(cols))
    th1 = template(None, "$\\Delta R$ Distribution Between Reconstructed \n Neutrinos and Matched Objects " + Fancy(name), None, name, base)
    th1.Histograms = sorted([top_correct, top_swp_b, top_swp_l, top_swp_bl], reverse = True, key = lambda x: sum(x.counts))
    th1.Filename = "deltaR-nunu"
    th1.SaveFigure()

    cols = colors()
    top_correct = template(make(exp.correct   , get_rtop, get_mass), Fancy(name, "sets"  ), next(cols))
    top_swp_b   = template(make(exp.swapped_bs, get_rtop, get_mass), Fancy(name, "swp_b" ), next(cols))
    top_swp_l   = template(make(exp.swapped_ls, get_rtop, get_mass), Fancy(name, "swp_l" ), next(cols))
    top_swp_bl  = template(make(exp.swapped_bl, get_rtop, get_mass), Fancy(name, "swp_bl"), next(cols))
    th1 = template(None, "Top Reconstruction (before $\\Delta R$ Minimization) " + Fancy(name), None, name, para_)
    th1.Histograms = sorted([top_correct, top_swp_b, top_swp_l, top_swp_bl], reverse = True, key = lambda x: sum(x.counts))
    th1.Filename = "nunu-before-deltaR"
    th1.SaveFigure()

    cols = colors()
    top_correct = template(make_atm(exp.correct   , fix_dr_nus, make_deltaR), Fancy(name, "sets"  ), next(cols))
    top_swp_b   = template(make_atm(exp.swapped_bs, fix_dr_nus, make_deltaR), Fancy(name, "swp_b" ), next(cols))
    top_swp_l   = template(make_atm(exp.swapped_ls, fix_dr_nus, make_deltaR), Fancy(name, "swp_l" ), next(cols))
    top_swp_bl  = template(make_atm(exp.swapped_bl, fix_dr_nus, make_deltaR), Fancy(name, "swp_bl"), next(cols))
    th1 = template(None, "$\\Delta R$ Minimization Between Reconstructed \n Neutrinos and Matched Objects " + Fancy(name), None, name, base)
    th1.Histograms = sorted([top_correct, top_swp_b, top_swp_l, top_swp_bl], reverse = True, key = lambda x: sum(x.counts))
    th1.Filename = "deltaR-nunu-minimization"
    th1.SaveFigure()

    cols = colors()
    top_correct = template(make_atm(exp.correct   , fix_dr_nus, get_dr_mass), Fancy(name, "sets"  ), next(cols))
    top_swp_b   = template(make_atm(exp.swapped_bs, fix_dr_nus, get_dr_mass), Fancy(name, "swp_b" ), next(cols))
    top_swp_l   = template(make_atm(exp.swapped_ls, fix_dr_nus, get_dr_mass), Fancy(name, "swp_l" ), next(cols))
    top_swp_bl  = template(make_atm(exp.swapped_bl, fix_dr_nus, get_dr_mass), Fancy(name, "swp_bl"), next(cols))
    th1 = template(None, "Top Reconstruction (after $\\Delta R$ Minimization) " + Fancy(name), None, name, para_)
    th1.Histograms = sorted([top_correct, top_swp_b, top_swp_l, top_swp_bl], reverse = True, key = lambda x: sum(x.counts))
    th1.Filename = "nunu-after-deltaR"
    th1.SaveFigure()

def combinatorial(sel):
    types = [
            ("top_children", False),
            ("truthjet"    , False),
            ("jetchildren" , False),
            ("jetleptons"  , False),
            ("top_children",  True),
            ("truthjet"    ,  True),
            ("jetchildren" ,  True),
            ("jetleptons"  ,  True)
    ]
    con = compiler(sel, types)

    para = {
        "xTitle" : "Invariant Mass for Leptonic Tops (GeV)", 
        "yTitle" : "Reconstructed Leptonic Tops / <units> (GeV)", 
        "Alpha" : 0.25, "xMax"   : 400, "xMin": 0, "xStep" : 40, "xBins" : 400, 
    }
    typx = ["top_children", "truthjet", "jetchildren", "jetleptons"]
    routine(con, typx, para, TopMassNominal, TopMassChi2)
    routine(con, typx, para, TopMassUnswapped, TopMassUnswappedChi2)
    routine(con, typx, para, NeutrinoAsymmetry, None)
    routine(con, typx, para, DeltaR, None)
    statistics(con)

