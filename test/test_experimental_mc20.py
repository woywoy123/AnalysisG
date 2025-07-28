from AnalysisG import Analysis
from AnalysisG.events import ExpMC20, BSM4Tops
from AnalysisG.core.io import IO

#root1 = "/home/tnom6927/Downloads/Binbin/dR0p05/user.bdong.510184.MGPy8EG.DAOD_PHYS.e8307_s3797_r13167_p6117.mc20a_truth_dR0p05_SSML_v01_output_root/user.bdong.38072717._000003.output.root"
#root1 = "./samples/mc20-experimental/*"
t = BSM4Tops()
root1 = "./samples/dilepton/DAOD_TOPQ1.29078691._000035.root"
ana = Analysis()
ana.AddSamples(root1, "tmp")
ana.AddEvent(t, "tmp")
ana.Threads = 1
ana.DebugMode = True
ana.Start()


#x = IO([root1])
#x.Trees = ["reco"]
#x.Leaves = ["el_charge"]
#for i in x:
#    print(i)


