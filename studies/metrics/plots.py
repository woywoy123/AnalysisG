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
    tl.LineWidth = 1

def MakeFigure(prf):
    name  = prf.ModeName
    name += ": " + prf.ModeName
    name += " kFold: " + str(prf.kFold)
    name += " Epoch: " + str(prf.Epoch)

    thx = TH1F()  
    thx.Title = name 

    dt = prf.RawTruth
    labl = {i : dt[i]["masses"] for i in dt}
    lng = {i : len(labl[i]) for i in labl} 

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
#    thx.ErrorBars = True
    thx.OutputDirectory = "Models/" + prf.ModelName + "/" + prf.ModeName
    thx.Filename = "epoch-" + str(prf.Epoch) + ".kfold-" + str(prf.kFold)
    thx.SaveFigure()


