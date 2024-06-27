from AnalysisG.generators.analysis import Analysis
from AnalysisG.events.event_bsm_4tops import BSM4Tops
from AnalysisG.core.selection_template import SelectionTemplate
from AnalysisG.selections.mc16.topkinematics.topkinematics import TopKinematics


smpls = "../../test/samples/dilepton/*"


ev = BSM4Tops()
sel = TopKinematics()

ana = Analysis()
ana.AddSamples(smpls, "tmp")
ana.AddEvent(ev, "tmp")
ana.AddSelection(sel)
ana.Start()


print(sel.res_top_kinematics)
print(sel.spec_top_kinematics)
print(sel.mass_combi)
print(sel.deltaR)

