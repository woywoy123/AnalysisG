from AnalysisG.core.plotting import TH1F, TH2F

def path(hist):
    hist.UseLateX = False
    hist.Style = "ATLAS"
    hist.OutputDirectory = "./output"
    hist.xTitle = "Invariant Mass of Candidate Top (GeV)"
    hist.yTitle = "Tops / ($2$ GeV)"
    hist.xMin   = 0
    hist.yMin   = 0
    hist.xStep  = 20
    hist.xMax   = 400
    hist.xBins  = 200
    hist.Stacked   = False
    hist.Overflow  = False
    hist.ShowCount = True
    return hist

def revert_mapping(data):
    pred_mass = data.p_topmass
    scr_mass  = data.prob_tops
    mass  = {}
    score = {}
    for i in pred_mass:
        v_mass = pred_mass[i]
        s_mass = scr_mass[i]
        for h in v_mass:
            if h not in score: mass[h] = []; score[h] = []
            mass[h]  += v_mass[h]
            score[h] += s_mass[h]
    return score, mass


def entry(ttbar, tttt, meta):
    tttt_score, tttt_mass = revert_mapping(tttt)
    ttbar_score, ttbar_mass = revert_mapping(ttbar)


    th2t = TH1F()
    th2t.Color = "red"
    th2t.Density = True
    th2t.ErrorBars = True
    th2t.xData = [i for i in sum(ttbar_score.values(), []) if i < 0.99]
    th2t.Title = r'$\bar{t}t$'

    th4t = TH1F()
    th4t.Color = "blue"
    th4t.Density = True
    th4t.ErrorBars = True
    th4t.xData = [i for i in sum(tttt_score.values(), []) if i < 0.99]
    th4t.Title = r'$\bar{t}t\bar{t}t$'

    thpx = TH1F()
    thpx.Histograms = [th2t, th4t]
    #thpx.Density = True
    thpx.Title = "Page Rank Scores of Reconstructed Top Candidates"
    thpx.xTitle = "Page Rank Score of Candidate Tops (Arb.)"
    thpx.yTitle = "Normalized by the number of Top Candidates (Arb.)"
    thpx.xMin = 0
    thpx.xMax = 1.01
    thpx.xBins = 200
    thpx.xStep = 0.1
    thpx.Overflow = False
    thpx.Style = "ATLAS"
    thpx.Filename = "pagerank"
    thpx.OutputDirectory = "./output/"
    thpx.SaveFigure()


    ttbar_msc = sum(ttbar_score.values(), [])
    tttt_msc  = sum(tttt_score.values(), [])

    ttbar_mx = sum(ttbar_mass.values(), [])
    tttt_mx  = sum(tttt_mass.values(), [])

    truth  = sum([sum(ttbar.t_topmass[i].values(), []) for i in ttbar.t_topmass], [])
    truth += sum([sum(tttt.t_topmass[i].values(), []) for i in tttt.t_topmass], [])

    tht = TH1F()
    tht.Color = "orange"
    tht.xData = truth
    tht.Title = 'Truth Tops'

    th2t = TH1F()
    th2t.Color = "red"
    th2t.xData = ttbar_mx
    th2t.Title = r'$\bar{t}t$'

    th4t = TH1F()
    th4t.Color = "blue"
    th4t.xData = tttt_mx
    th4t.Title = r'$\bar{t}t\bar{t}t$'

    thpx = TH1F()
    thpx.Histogram = tht
    thpx.Stacked = True
    thpx.Histograms = [th2t, th4t]
    thpx.Title = "Unweighted Invariant Mass of Reconstructed Top Candidates"
    thpx.xTitle = "Invariant Mass of Top Candidate (GeV)"
    thpx.yTitle = "Number of Top Candidates / (1 GeV)"
    thpx.xMin = 80
    thpx.xMax = 300
    thpx.xBins = 220
    thpx.xStep = 20
    thpx.Overflow = False
    thpx.Style = "ATLAS"
    thpx.Filename = "top-mass"
    thpx.OutputDirectory = "./output/"
    thpx.SaveFigure()

    thpx = TH2F()
    thpx.Title = "PageRank Score as a Function of Top Candidate Invariant Mass ($t\\bar{t}$)"
    thpx.xTitle = "Invariant Mass of Top Candidate (GeV)"
    thpx.yTitle = "Page Rank Score of Top Candidate (Arb.)"

    thpx.xMin  = 80
    thpx.xMax  = 300
    thpx.xBins = 220
    thpx.xStep = 20

    thpx.yStep = 0.1
    thpx.yBins = 100
    thpx.yMax = 1.0
    thpx.yMin = 0

    thpx.Color = "magma"
    thpx.xData = [ttbar_mx[i]  for i in range(len(ttbar_msc)) if ttbar_msc[i] < 0.95]
    thpx.yData = [ttbar_msc[i] for i in range(len(ttbar_msc)) if ttbar_msc[i] < 0.95]
    thpx.Style = "ATLAS"
    thpx.Filename = "top-mass-ttbar"
    thpx.OutputDirectory = "./output/"
    thpx.SaveFigure()


    thpx = TH2F()
    thpx.Title = "PageRank Score as a Function of Top Candidate Invariant Mass ($t\\bar{t}t\\bar{t}$)"
    thpx.xTitle = "Invariant Mass of Top Candidate (GeV)"
    thpx.yTitle = "Page Rank Score of Top Candidate (Arb.)"

    thpx.xMin  = 80
    thpx.xMax  = 300
    thpx.xBins = 220
    thpx.xStep = 20

    thpx.yStep = 0.1
    thpx.yBins = 100
    thpx.yMax = 1.0
    thpx.yMin = 0

    thpx.Color = "magma"
    thpx.xData = [tttt_mx[i]  for i in range(len(tttt_msc)) if tttt_msc[i] < 0.95]
    thpx.yData = [tttt_msc[i] for i in range(len(tttt_msc)) if tttt_msc[i] < 0.95]
    thpx.Style = "ATLAS"
    thpx.Filename = "top-mass-tttt"
    thpx.OutputDirectory = "./output/"
    thpx.SaveFigure()





