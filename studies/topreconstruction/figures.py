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
    tttt_mass, tttt_score = revert_mapping(tttt)
    ttbar_mass, tttbar_score = revert_mapping(ttbar)

    th2t = TH1F()
    th2t.Color = "red"
    th2t.Density = True
    th2t.ErrorBars = True
    th2t.xData = [i for i in sum(ttbar_mass.values(), []) if i < 0.99]
    th2t.Title = r'$\bar{t}t$'

    th4t = TH1F()
    th4t.Color = "blue"
    th4t.Density = True
    th4t.ErrorBars = True
    th4t.xData = [i for i in sum(tttt_mass.values(), []) if i < 0.99]
    th4t.Title = r'$\bar{t}t\bar{t}t$'

    thpx = TH1F()
    thpx.Histograms = [th2t, th4t]
    #thpx.Density = True
    thpx.Title = "Page Rank Scores of Reconstructed Top Quarks"
    thpx.xTitle = "Page Rank Score (Arb.)"
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








#    ttbar_tru_mass  = ttbar.t_topmass
#    tttt_tru_mass  = tttt.t_topmass
#





#    thpx = TH1F()
#    thpx.xData = sum([sum(top_scr[i].values(), []) for i in top_scr], [])
#    thpx.xMin = 0
#    thpx.xMax = 1
#    thpx.xBins = 1000
#    thpx.yMax = 200
#    thpx.xStep = 0.1
#    thpx.xTitle = "Page Rank Score (Arb.)"
#    thpx.yTitle = "Top Candidates (Arb.)"
#    thpx.Title = "Page Rank Scores of Reconstructed Top Quarks"
#    thpx.Filename = "pagerank"
#    thpx.OutputDirectory = "./output/"
#    thpx.Style = "ATLAS"
#    thpx.Overflow = False
#    thpx.SaveFigure()
#
#
#    reco = []
#    truth = []
#
#    for i in pred_mass: reco += sum(pred_mass[i].values(), [])
#    for i in tru_mass: truth += sum(tru_mass[i].values(), [])
#
#    thp = TH1F()
#    thp.Title = "Reconstructed"
#    thp.Color = "red"
#    thp.xData = reco
#
#    tht = TH1F()
#    tht.Title = "Truth"
#    tht.Color = "blue"
#    tht.xData = truth
#
#    th = path(TH1F())
#    th.Title = "Reconstructed Tops Compared to Truth"
#    th.Histograms = [thp, tht]
#    th.Filename = "tops-reco"
#    th.SaveFigure()







