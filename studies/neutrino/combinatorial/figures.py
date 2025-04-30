from AnalysisG.core.plotting import TH1F, TH2F
from .helpers import compiler 
from .utils import *
import pandas

typx = ["top_children", "truthjet", "jetchildren", "jetleptons"]
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

def make(data_, fx):
    out = []
    for k in range(2): out += [data_[k][j] for j in data_[k]]
    return fx(sum(out, []))

def TopMasses(con):
    def get_top_mass(inpt): return [i.truthtop for i in inpt if i if not None and i.truthtop is not None]

    para = {
        "xTitle" : "Invariant Mass for Leptonic Tops (GeV)", 
        "yTitle" : "Density (Arb.) / <units> (GeV)", 
        "Alpha" : 0.25, 
        "xMax"   : 400, "xMin": 0, "xStep" : 40, "xBins" : 400,
        "Density" : False,
    }
    for i in typx:
        cols = colors()
        data  = con.get(i, False)
        top_sets   = template(make(data.sets      , get_top_mass), Fancy(i, "sets"  ), next(cols))
        top_swp_b  = template(make(data.swapped_bs, get_top_mass), Fancy(i, "swp_b" ), next(cols))
        top_swp_l  = template(make(data.swapped_ls, get_top_mass), Fancy(i, "swp_l" ), next(cols))
        top_swp_bl = template(make(data.swapped_bl, get_top_mass), Fancy(i, "swp_bl"), next(cols))
        top_fakes  = template(make(data.fake_nus  , get_top_mass), Fancy(i, "fake"  ), next(cols))
        th1 = template(None, "Double Neutrino Reconstruction Performance " + Fancy(i), None, i, para)
        th1.Stacked = True
        th1.Histograms = sorted([top_sets, top_swp_b, top_swp_l, top_swp_bl, top_fakes], key = lambda x: sum(x.counts))
        th1.Filename = "all_nunu"
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
    TopMasses(con)
    exit()
    statistics(con)

