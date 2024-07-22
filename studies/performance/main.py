from AnalysisG.generators.analysis import Analysis

# event implementation
from AnalysisG.events.bsm_4tops.event_bsm_4tops import BSM4Tops

# study
from AnalysisG.selections.performance.topefficiency.topefficiency import TopEfficiency

# figures
import topefficiency
import pickle

study = "topefficiency"
smpls = "../../test/samples/dilepton/*"

plotting_method = {
    "topefficiency" : topefficiency
}

gen_data = True
figure_path = "./Output/"
for mass in ["1000", "900", "800", "700", "600", "500", "400"]:
    mass_point = "Mass." + mass + ".GeV"
    smpls = "/home/tnom6927/Downloads/DileptonCollection/MadGraphPythia8EvtGen_noallhad_ttH_tttt_m" + mass + "/*"

    method = plotting_method[study]
    method.figures.figure_path = figure_path
    method.figures.mass_point  = mass_point

    if gen_data:
        ev = BSM4Tops()

        sel = None
        if study == "topefficiency": sel = TopEfficiency()

        ana = Analysis()
        ana.AddSamples(smpls, "tmp")
        ana.AddEvent(ev, "tmp")
        ana.AddSelection(sel)
        ana.Start()

        f = open(study + "-" + mass_point + ".pkl", "wb")
        pickle.dump(sel, f)
        f.close()

    f = open(study + "-" + mass_point + ".pkl", "rb")
    pres = pickle.load(f)
    f.close()

    print("plotting: " + study)
    if study == "topefficiency": method.figures.TopEfficiency(pres)
