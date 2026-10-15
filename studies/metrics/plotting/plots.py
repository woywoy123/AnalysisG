from AnalysisG.core import TH1F, TH2F
from AnalysisG.core import Tools
import pickle

def default(tl):
    tl.Style = "ATLAS"
    tl.DPI = 300
    tl.TitleSize = 15
    tl.AutoScaling = True
    tl.LegendSize = 10
    tl.yScaling = 5
    tl.xScaling = 10
    tl.FontSize = 10
    tl.AxisSize = 10
    try: tl.LineWidth = 0.1
    except: pass
    return tl

def templ_th1f(
    name, xbins, xmin, xmax, xdata, 
    xtitle, ytitle, hists = None, colrs = None
):
    tf = default(TH1F())
    tf.Title = name
    tf.xData = xdata;   tf.xBins  = xbins
    tf.xTitle = xtitle; tf.yTitle = ytitle
    tf.xMin   = xmin;   tf.xMax   = xmax
    if hists is None: return tf
    tf.Histograms = hists
    if colrs is None: return tf
    for i in hists: i.Color = next(colrs)
    return tf

def basic_th1f(name, data):
    tf = TH1F()
    tf.Title = name
    tf.xData = data
    return tf

def basic_th2f(name, xdata, ydata):
    tf = TH2F()
    tf.Title = name
    tf.xData = xdata
    tf.yData = ydata
    return tf

def MakeFigure(prf):
    name  = prf.ModeName
    name += ": " + prf.ModeName
    name += " kFold: " + str(prf.kFold)
    name += " Epoch: " + str(prf.Epoch)

    thx = TH1F()  
    thx.Title = name 

    dt   = prf.RawTruth
    labl = {i : dt[i]["masses"] for i in dt}
    lng  = {i : len(labl[i]) for i in labl} 

    for i in labl:
        th = TH1F()  
        th.Title = i.decode("utf-8")
        th.xData = labl[i]
        thx.Histograms.append(th)
    
    thx.xTitle = "Mass (GeV)"
    thx.yTitle = "Truth Tops"
    thx.Stacked = True
    thx.xBins = 500
    thx.xStep = 50
    thx.xMin  = 0
    thx.xMax  = 500
    thx.OutputDirectory = "Models/" + prf.ModelName + "/" + prf.ModeName
    thx.Filename = "epoch-" + str(prf.Epoch) + ".kfold-" + str(prf.kFold)
    thx.SaveFigure()



