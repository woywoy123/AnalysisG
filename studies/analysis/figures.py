from AnalysisG.core.plotting import TH1F

def template_hist(title, weights, data, bins, xmax, xtitle, xmin = None, xstep = None, fname = None):
    f = TH1F()
    f.Title = title
    f.Weights = weights
    f.xData = data
    f.xBins = bins
    f.xMax = xmax
    if xmin is not None: f.xMin = xmin
    if xstep is not None: f.xStep = xstep
    f.Color = "orange"
    f.yTitle = "Events"
    f.xTitle = xtitle
    f.Filename = fname if fname is not None else title
    f.Overflow = "none"
    f.ShowCount = True
    f.ErrorBars = True
    return f

def GetValues(sel, key):
    w  = [i[key]["weight"]    for i in sel.output if i[key]["passed"]]
    v1 = [i[key]["variable1"] for i in sel.output if i[key]["passed"]]
    v2 = [i[key]["variable2"] for i in sel.output if i[key]["passed"]]
    return w, v1, v2

def CRttbarCO2l_CO(sel):
    w, v1, v2 = GetValues(sel, "CRttbarCO2l_CO")
    th = template_hist("CRttbarCO2l_CO", w, v1, [0, 1, 2], 2, "passed")
    th.SaveFigure()

def CRttbarCO2l_CO_2b(sel):
    w, v1, v2 = GetValues(sel, "CRttbarCO2l_CO_2b")
    th = template_hist("CRttbarCO2l_CO_2b", w, v1, [0, 1, 2], 2, "passed")
    th.SaveFigure()

def CRttbarCO2l_gstr(sel):
    w, v1, v2 = GetValues(sel, "CRttbarCO2l_gstr")
    th = template_hist("CRttbarCO2l_gstr", w, v1, [0, 1, 2], 2, "passed")
    th.SaveFigure()

def CRttbarCO2l_gstr_2b(sel):
    w, v1, v2 = GetValues(sel, "CRttbarCO2l_gstr_2b")
    th = template_hist("CRttbarCO2l_gstr_2b", w, v1, [0, 1, 2], 2, "passed")
    th.SaveFigure()

def CR1b3lem(sel):
    w, v1, v2 = GetValues(sel, "CR1b3lem")
    th = template_hist("CR1b3lem", w, v1, [0.0, 15.0, 30.0, 50.0], 50, "lep2_pt")
    th.SaveFigure()

def CR1b3le(sel):
    w, v1, v2 = GetValues(sel, "CR1b3le")
    th = template_hist("CR1b3le", w, v1, [0.0, 15.0, 30.0, 50.0], 50, "lep2_pt")
    th.SaveFigure()

def CR1b3lm(sel):
    w, v1, v2 = GetValues(sel, "CR1b3lm")
    th = template_hist("CR1b3lm", w, v1, [0.0, 15.0, 30.0, 50.0], 50, "lep2_pt")
    th.SaveFigure()

def CRttW2l_plus(sel):
    w, v1, v2 = GetValues(sel, "CRttW2l_plus")
    th = template_hist("CRttW2l_plus", w, v1, 7, 7, "nJets", 0, 1)
    th.SaveFigure()

def CRttW2l_minus(sel):
    w, v1, v2 = GetValues(sel, "CRttW2l_minus")
    th = template_hist("CRttW2l_minus", w, v1, 7, 7, "nJets", 0, 1)
    th.SaveFigure()

def CR1bplus(sel):
    w, v1, v2 = GetValues(sel, "CR1bplus")
    th = template_hist("CR1bplus", w, v1, 11, 11, "nJets", 0, 1)
    th.SaveFigure()

def CR1bminus(sel):
    w, v1, v2 = GetValues(sel, "CR1bminus")
    th = template_hist("CR1bminus", w, v1, 11, 11, "nJets", 0, 1)
    th.SaveFigure()

def CRttW2l(sel):
    w, v1, v2 = GetValues(sel, "CRttW2l")
    th = template_hist("CRttW2l", w, v1, [70, 80, 100, 120, 140, 160, 180, 240, 300], 300.0, "lep_sum_pt", 70, 20)
    th.SaveFigure()

def VRttZ3l(sel):
    return 

def VRttWCRSR(sel):
    return 

def SR4b(sel):
    return 

def SR2b(sel):
    return 

def SR3b(sel):
    return 

def SR2b2l(sel):
    return 

def SR2b3l4l(sel):
    return 

def SR2b4l(sel):
    return 

def SR3b2l(sel):
    return 

def SR3b3l4l(sel):
    return 

def SR3b4l(sel):
    return 

def SR4b4l(sel):
    return 

def SR(sel):
    return 

def entry(sel):
#    CRttbarCO2l_CO(sel)
#    CRttbarCO2l_CO_2b(sel)
#    CRttbarCO2l_gstr(sel)
#    CRttbarCO2l_gstr_2b(sel)
#    CR1b3lem(sel)
#    CR1b3le(sel)
#    CR1b3lm(sel)
#    CRttW2l_plus(sel)
#    CRttW2l_minus(sel)
#    CR1bplus(sel)
#    CR1bminus(sel)
    CRttW2l(sel)
    VRttZ3l(sel)
    VRttWCRSR(sel)
    SR4b(sel)
    SR2b(sel)
    SR3b(sel)
    SR2b2l(sel)
    SR2b3l4l(sel)
    SR2b4l(sel)
    SR3b2l(sel)
    SR3b3l4l(sel)
    SR3b4l(sel)
    SR4b4l(sel)
    SR(sel)
