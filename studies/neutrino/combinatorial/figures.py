from AnalysisG.core.plotting import TH1F, TH2F
from .helpers import compiler 
import pandas

typx = ["topchildren", "truthjet", "jetchildren", "jetleptons"]
def statistics(data, eps = None):
    losses = {i : loss(eps) for i in typx}
    for i in range(len(data["topchildren"])):
        if data["topchildren"][i] is None: continue
        for k in typx: losses[k].add(data[k][i])
    return losses


def colors():
    return iter(sorted(list(set([
        "aqua", "orange", "green","blue","olive","teal","gold",
        "darkblue","azure","lime","crimson","cyan","magenta","orchid",
        "sienna","silver","salmon","chocolate", "navy", "plum", "indigo", 
        "lightblue", "lavender", "ivory"
    ]))))


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
    if pth is not None: hist.OutputDirectory = "figures/" + pth
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



def TopMasses(con):
    para = {
        "xTitle" : "Invariant Mass for Leptonic Tops (GeV)", 
        "yTitle" : "Density (Arb.) / <units> (GeV)", 
        "Alpha" : 0.25, 
        "xMax"   : 400, "xMin": 0, "xStep" : 40, "xBins" : 400,
        "Density" : False,
    }
    for i in typx:
        data    = con.get(i, False)

        swp_ls  = data.swp_ls
        swp_bs  = data.swp_bs
        swp_lb  = data.swp_lb
        swp_alb = data.swp_alb









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
    statistics(con)

