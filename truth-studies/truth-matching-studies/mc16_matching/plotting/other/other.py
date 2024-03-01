from AnalysisG.Plotting import TH1F, TH2F

def settings():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : "./plt_plots/other/",
            "Histograms" : [],
            "Histogram" : None,
            "LegendLoc" : "upper right"
    }
    return settings

def deltaR_lepB(ana):

    sett = settings()
    th_delqCT = TH1F(**sett)
    th_delqCT.xData = ana.deltaR_lepquark["CT"]
    th_delqCT.Title = "Correct Top"

    th_delqFT = TH1F(**sett)
    th_delqFT.xData = ana.deltaR_lepquark["FT"]
    th_delqFT.Title = "False Top"

    th = TH1F(**sett)
    th.Histograms = [th_delqCT, th_delqFT]
    th.Title = "$\\Delta$R Between Lepton and B-Quark"
    th.xTitle = "$\\Delta$R"
    th.yTitle = "Entries (arb.)"
    th.xBins = 100
    th.xStep = 0.2
    th.xScaling = 20
    th.yScaling = 10
    th.xMin  = 0
    th.xMax  = 5
    th.Filename = "Figure.X.a"
    th.SaveFigure()


def reconstruction_algorithm(ana):
    nunu_data = ana.nunu_data
    nevents = ana.TotalEvents
    nevents_final = ana.CutFlow["Strategy::::Ambiguous"]
    eff_closestLeptonicGroup   = sum(ana.eff_closestLeptonicGroup)
    eff_remainingLeptonicGroup = sum(ana.eff_remainingLeptonicGroup)
    eff_bestHadronicGroup      = sum(ana.eff_bestHadronicGroup)
    eff_remainingHadronicGroup = sum(ana.eff_remainingHadronicGroup)
    eff_resonance_had          = sum(ana.eff_resonance_had)
    eff_resonance_lep          = sum(ana.eff_resonance_lep)
    eff_resonance              = sum(ana.eff_resonance)

    string = ""
    string += f"Number of events passed: {nevents_final} / {nevents} \n"
    string += "Efficiencies: \n"
    string += f"Closest leptonic group from same top: {eff_closestLeptonicGroup / (nevents_final)} \n"
    string += f"Remaining leptonic group from same top: {eff_remainingLeptonicGroup / (nevents_final)} \n"
    string += f"Closest hadronic group from same top: {eff_bestHadronicGroup / (nevents_final)} \n"
    string += f"Remaining hadronic group from same top: {eff_remainingHadronicGroup / (nevents_final)} \n"
    string += f"Leptonic decay products correctly assigned to resonance: {eff_resonance_lep / (nevents_final)} \n"
    string += f"Hadronic decay products correctly assigned to resonance: {eff_resonance_had / (nevents_final)} \n"
    string += f"All decay products correctly assigned to resonance: {eff_resonance / (nevents_final)} \n"
    print(string)


    # Plotting 
    sett = settings()
    hist = []
    for i in [["had_res", "Hadronic Resonant"], ["had_spec", "Hadronic Spectator"]]:
        th = TH1F()
        th.xData = ana.nunu_data[i[0]]
        th.Title = i[1]
        hist.append(th)

    th = TH1F(**sett)
    th.Histograms = hist
    th.Title = "Reconstructed Hadronic Top Mass"
    th.xTitle = "Mass (GeV)"
    th.yTitle = "Entries <unit>"
    th.xBins = 200
    th.xMin = 0
    th.xMax = 250
    th.xStep = 50
    th.OverFlow = True
    th.Filename = "Figure.X.b"
    th.SaveFigure()


    hist = []
    for i in [["lep_res", "Leptonic Resonant"], ["lep_spec", "Leptonic Spectator"]]:
        th = TH1F()
        th.xData = ana.nunu_data[i[0]]
        th.Title = i[1]
        hist.append(th)

    th = TH1F(**sett)
    th.Histograms = hist
    th.Title = "Reconstructed Leptonic Top Mass"
    th.xTitle = "Mass (GeV)"
    th.yTitle = "Entries <unit>"
    th.xBins = 200
    th.xMin = 0
    th.xMax = 250
    th.xStep = 50
    th.OverFlow = True
    th.Filename = "Figure.X.c"
    th.SaveFigure()


    th = TH1F(**sett)
    th.xData = ana.nunu_data["res_mass"]
    th.Title = "Reconstructed Leptonic Top Mass"
    th.xTitle = "Mass (GeV)"
    th.yTitle = "Entries <unit>"
    th.xBins = 150
    th.xMin = 0
    th.xMax = 1500
    th.xStep = 150
    th.OverFlow = True
    th.Filename = "Figure.X.d"
    th.SaveFigure()

    th = TH1F(**sett)
    th.xData = ana.nunu_data["num_sols"]
    th.Title = "Number of Dilepton Reconstruction Solutions"
    th.xTitle = "Number of neutrino solutions"
    th.yTitle = "Number"
    th.xMin = 0
    th.xStep = 1
    th.xBinCentering = True
    th.Filename = "Figure.X.e"
    th.SaveFigure()






def AddOnStudies(ana):
    deltaR_lepB(ana)
    reconstruction_algorithm(ana)


