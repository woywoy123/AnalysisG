from AnalysisG import Analysis
from AnalysisG.events.exp_mc20 import ExpMC20
from AnalysisG.events.ssml_mc20 import SSML_MC20
from AnalysisG.events.bsm_4tops import BSM4Tops
from AnalysisG.selections.mc20.matching.matching import TopMatching
from figures import entry
import pathlib
import pickle

modes = {"BSM_4TOPS" : BSM4Tops, "SSML_MC20" : SSML_MC20, "EXP_MC20": ExpMC20}

exp_mc20  = "./samples/exp/output.root"

#ssml_mc20 = "/home/tnom6927/Downloads/mc20/current/mc20_13TeV.412043.aMcAtNloPythia8EvtGen_A14NNPDF31_SM4topsNLO.deriv.DAOD_PHYS.e7101_a907_r14859_p6490/user.rqian.42181793._000001.output.root"
#ssml_mc16 = "/home/tnom6927/Downloads/mc20/equivalent-mc16/user.tnommens.412043.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e7101_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root/user.tnommens.40945849._000001.output.root"
smpls = {
        "EXP_MC20"  : exp_mc20,
#        "SSML_MC20" : ssml_mc20, 
#        "BSM_4TOPS" : ssml_mc16, 
}

for evnt in smpls:
    sel = TopMatching() 
    ev = modes[evnt]()
    ana = Analysis()
    ana.FetchMeta = False
#    ana.Threads = 2
    ana.DebugMode = True
    ana.AddSamples(smpls[evnt], "dr")
    ana.AddEvent(ev, "dr")
    ana.AddSelection(sel)
    ana.SaveSelectionToROOT = True
    ana.Start()

exit()
entry(smpls, "./ProjectName/Selections/matching-")







