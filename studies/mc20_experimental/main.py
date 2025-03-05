from AnalysisG import Analysis
from AnalysisG.core.io import IO
from AnalysisG.events.exp_mc20 import ExpMC20
from AnalysisG.events.ssml_mc20 import SSML_MC20
from AnalysisG.selections.mc20_compare.topmatching_fuzzy.mc20_fuzzy import TopMatchingFuzzy
from AnalysisG.selections.mc20_compare.topmatching_current.mc20_cur import TopMatchingCurrent
from figures import entry

#smpls = "/home/tnom6927/Downloads/mc20/current/mc20_13TeV.412043.aMcAtNloPythia8EvtGen_A14NNPDF31_SM4topsNLO.deriv.DAOD_PHYS.e7101_a907_r14859_p6490/user.rqian.42181793._000001.output.root"
smpls = "./output.root"

ev = ExpMC20() #SSML_MC20()
sel = TopMatchingFuzzy() #TopMatchingCurrent()
selx = sel.load()
if selx is None:
    ana = Analysis()
    ana.FetchMeta = False
    ana.Threads = 12
    ana.AddSamples(smpls, "dr")
    ana.AddEvent(ev, "dr")
    ana.AddSelection(sel)
    ana.Start()
    sel.dump()
    selx = sel

print("plotting")
entry(selx)







