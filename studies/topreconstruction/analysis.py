from AnalysisG.core.plotting import TH1F
from styles import *

def entry(mx):
    
    # --------- check the general classification performance ------ #
#    hist = {}
#    for prc in mx.output:
#        hist[prc] = TH1F()
#        hist[prc].Title = prc
#        hist[prc].Density = True
#        hist[prc].xData = mx.output[prc].truth.truth_mass 
#
#    thpx = TH1F()
#    thpx.Stacked = True
#    thpx.Histograms = [i for i in hist.values()]
#    thpx.Title = "Unweighted Invariant Mass of Truth Tops Candidates"
#    thpx.xTitle = "Invariant Mass of Top Candidate (GeV)"
#    thpx.yTitle = "Number of Top Candidates / (1 GeV)"
#    thpx.xMin = 80
#    thpx.xMax = 300
#    thpx.xBins = 220
#    thpx.xStep = 20
#    thpx.Overflow = False
#    thpx.Style = "ATLAS"
#    thpx.Filename = "top-mass"
#    thpx.OutputDirectory = "./output/"
#    thpx.SaveFigure()

    # -------- plot top mass distribution per Epoch -------- #
    pdata = {} 
    for prc in mx.output:
        for epc in mx.output[prc].epochs:
            for md in mx.output[prc].epochs[epc].nominals: 
                if epc not in pdata: pdata[epc] = {}
                if md not in pdata[epc]: pdata[epc][md] = {}
                if prc not in pdata[epc][md]: 
                    pdata[epc][md][prc] = {
                            "truth" : None, "nominal" : None, 
                            "Masked": None, "Unmasked" : None
                    }
                pdata[epc][md][prc]["nominal"] = mx.output[prc].epochs[epc].nominals[md]
                pdata[epc][md][prc]["truth"  ] = mx.output[prc].truth
                pdata[epc][md][prc]["masked" ] = mx.output[prc].epochs[epc].masked[md]
                pdata[epc][md][prc]["unmasked"] = mx.output[prc].epochs[epc].unmasked[md]

    for epc in pdata:
        for md in pdata[epc]:
            hist = []
            for prc in pdata[epc][md]:
                hist.append(TH1F())
                hist[-1].Title = prc + " (Nominal)"
                hist[-1].xData = pdata[epc][md][prc]["nominal"].kfolds_ntop

            for prc in pdata[epc][md]:
                hist.append(TH1F())
                hist[-1].Title = prc + " (Masked)"
                hist[-1].Hatch = "///"
                hist[-1].xData = pdata[epc][md][prc]["masked"].kfolds_ntop

            for prc in pdata[epc][md]:
                hist.append(TH1F())
                hist[-1].Title = prc + " (Unmasked)"
                hist[-1].Hatch = "\\\\"
                hist[-1].xData = pdata[epc][md][prc]["unmasked"].kfolds_ntop

            thpx = TH1F()
            thpx.Stacked = False
            thpx.Density = True
            thpx.Histograms = hist
            thpx.Title  = "Difference in N-Tops Clusters using Three Clustering Methods"
            thpx.xTitle = "Fraction of Truth Tops Recovered ($\\Delta \\text{top}_{\\text{tru}} - \\text{pred}$)"
            thpx.yTitle = "Number of Top Candidates"
            thpx.xMin = -4
            thpx.xMax =  4
            thpx.xBins = 8
            thpx.xStep = 1
            thpx.FX("chi2")
            thpx.Overflow = False
            thpx.Style = "ATLAS"
            thpx.Filename = md
            thpx.OutputDirectory = "./output/" + str(epc)
            thpx.SaveFigure()

            hist = []
            for prc in pdata[epc][md]:
                hist.append(TH1F())
                hist[-1].Title = prc + " (Nominal)"
                hist[-1].xData = [i**0.5 for i in pdata[epc][md][prc]["nominal"].kfolds_chi2]

            for prc in pdata[epc][md]:
                hist.append(TH1F())
                hist[-1].Title = prc + " (Masked)"
                hist[-1].Hatch = "///"
                hist[-1].xData = [i**0.5 for i in pdata[epc][md][prc]["masked"].kfolds_chi2]

            for prc in pdata[epc][md]:
                hist.append(TH1F())
                hist[-1].Title = prc + " (Unmasked)"
                hist[-1].Hatch = "\\\\"
                hist[-1].xData = [i**0.5 for i in pdata[epc][md][prc]["unmasked"].kfolds_chi2]

            thpx = TH1F()
            thpx.Stacked = False
            thpx.Histograms = hist
            thpx.Title  = "$\\chi$ matched Top Cluster"
            thpx.xTitle = "Lowest $\\chi$ Error of Cluster Top"
            thpx.yTitle = "Number of Top Candidates / (1 GeV)"
            thpx.xMin = 0
            thpx.xMax = 12000
            thpx.xBins = 500
            thpx.xStep = 1000
            thpx.FX("chi2")
            thpx.Overflow = True
            thpx.Style = "ATLAS"
            thpx.Filename = "clustering_chi2_" + md
            thpx.OutputDirectory = "./output/epoch-" + str(epc)
            thpx.SaveFigure()


            hist = []
            for prc in pdata[epc][md]:
                hist.append(TH1F())
                hist[-1].Title = prc + " (Truth)"
                hist[-1].Color = "black"
                hist[-1].xData = pdata[epc][md][prc]["truth"].truth_mass
            
            for prc in pdata[epc][md]:
                hist.append(TH1F())
                hist[-1].Title = prc + " (Nominal)"
                hist[-1].xData = pdata[epc][md][prc]["nominal"].top_mass

            for prc in pdata[epc][md]:
                hist.append(TH1F())
                hist[-1].Title = prc + " (Masked)"
                hist[-1].Hatch = "///"
                hist[-1].xData = pdata[epc][md][prc]["masked"].top_mass

            for prc in pdata[epc][md]:
                hist.append(TH1F())
                hist[-1].Title = prc + " (Unmasked)"
                hist[-1].Hatch = "\\\\"
                hist[-1].xData = pdata[epc][md][prc]["unmasked"].top_mass

            thpx = TH1F()
            thpx.Histograms = hist
            thpx.Alpha = 0.5 
            thpx.Title = "Unweighted Invariant Mass of Truth Tops Candidates"
            thpx.xTitle = "Invariant Mass of Top Candidate (GeV)"
            thpx.yTitle = "Number of Top Candidates / (1 GeV)"
            thpx.xMin = 0
            thpx.xMax = 400
            thpx.xBins = 400
            thpx.xStep = 20
            thpx.Overflow = False
            thpx.Style    = "ATLAS"
            thpx.Filename = "top_mass_" + md
            thpx.OutputDirectory = "./output/epoch-" + str(epc)
            thpx.SaveFigure()





















