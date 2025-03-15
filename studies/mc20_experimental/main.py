from AnalysisG import Analysis
from AnalysisG.core.io import IO
from AnalysisG.events.exp_mc20 import ExpMC20
from AnalysisG.events.ssml_mc20 import SSML_MC20
from AnalysisG.events.bsm_4tops import BSM4Tops
from AnalysisG.selections.mc20.matching.matching import TopMatching
from figures import entry
modes = {"BSM_4TOPS" : BSM4Tops, "SSML_MC20" : SSML_MC20, "EXP_MC20": ExpMC20}

#exp_mc20  = "/home/tnom6927/Downloads/mc20/fuzzy/output.root"
exp_mc20  = "./samples/exp/output.root"
ssml_mc20 = "./samples/ref/*" #"/home/tnom6927/Downloads/mc20/current/mc20_13TeV.412043.aMcAtNloPythia8EvtGen_A14NNPDF31_SM4topsNLO.deriv.DAOD_PHYS.e7101_a907_r14859_p6490/user.rqian.42181793._000001.output.root"
ssml_mc16 = "/home/tnom6927/Downloads/mc20/equivalent-mc16/user.tnommens.412043.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e7101_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root/user.tnommens.40945849._000001.output.root"
smpls = {
        "EXP_MC20"  : exp_mc20,
        "SSML_MC20" : ssml_mc20, 
        "BSM_4TOPS" : ssml_mc16, 
}

#for evnt in smpls:
#    sel = TopMatching() 
#    ev = modes[evnt]()
#    if sel.load("pkl-data", ev.Name) is not None: continue
#    ana = Analysis()
#    ana.FetchMeta = False
#    ana.Threads = 12
#    ana.AddSamples(smpls[evnt], "dr")
#    ana.AddEvent(ev, "dr")
#    ana.AddSelection(sel)
#    ana.Start()
#    sel.dump("pkl-data", ev.Name)

selx = TopMatching()
print("plotting")
data = [selx.load("pkl-data", modes[i]().Name) for i in smpls]
entry(*data)







