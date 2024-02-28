from AnalysisG.Plotting import TH2F, TH1F

def TemplateTH2F(Title, Data, f_title):
    th2 = TH2F()
    th2.Title = Title
    th2.xData = Data["mass"]
    th2.yData = Data["pt"]

    th2.xTitle = "Z/H-Prime Mass (GeV)"
    th2.yTitle = "Transverse Momentum of Z/H-Prime (GeV)"
    th2.Filename = "Figure.1." + f_title

    th2.xBins = 750
    th2.yBins = 500

    th2.xMin = 0
    th2.yMin = 0

    th2.xMax = 1500
    th2.yMax = 1000
    th2.yOverFlow = True
    th2.xOverFlow = True

    th2.OutputDirectory = "./plt_plots/resonance/"
    th2.SaveFigure()

def ZPrime(ana):
    TemplateTH2F("Z-Prime Mass-PT Matrix - Truth Tops", ana.zprime_mass_tops, "a")
    TemplateTH2F("Z-Prime Mass-PT Matrix - Truth Children", ana.zprime_mass_children, "b")
    TemplateTH2F("Z-Prime Mass-PT Matrix - Truth Jets", ana.zprime_mass_truthjets, "c")
    TemplateTH2F("Z-Prime Mass-PT Matrix - Jets", ana.zprime_mass_jets, "d")
