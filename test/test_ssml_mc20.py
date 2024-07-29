from AnalysisG.generators.analysis import Analysis
from AnalysisG.events import SSML_MC20
from AnalysisG.core.io import IO
from AnalysisG.graphs.ssml_mc20 import GraphJets

root1 = "/home/tnom6927/Downloads/mc20/user.bdong.510203.MGPy8EG.DAOD_PHYS.e8559_a911_r15224_p6266.DNN_withsys_v01_output/user.bdong.40403032._000001.output.root"

t = SSML_MC20()
gr = GraphJets()

ana = Analysis()
ana.AddSamples(root1, "tmp")
ana.AddEvent(t, "tmp")
ana.AddGraph(gr, "tmp")
ana.Start()
