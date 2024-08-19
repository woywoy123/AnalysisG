from AnalysisG.generators.analysis import Analysis
from AnalysisG.graphs.ssml_mc20 import GraphJets
from AnalysisG.events import SSML_MC20

from AnalysisG.core.lossfx import OptimizerConfig
from AnalysisG.models import RecursiveGraphNeuralNetwork

root1 = "/home/tnom6927/Downloads/mc20/user.bdong.510203.MGPy8EG.DAOD_PHYS.e8559_a911_r15224_p6266.DNN_withsys_v01_output/user.bdong.40403032._000001.output.root"

ev = SSML_MC20()
gr = GraphJets()
md = RecursiveGraphNeuralNetwork()
md.o_edge = {
        "top_edge" : "CrossEntropyLoss",
        "res_edge" : "CrossEntropyLoss"
}

md.o_graph = {
        "ntops"  : "CrossEntropyLoss",
        "signal" : "CrossEntropyLoss"
}
md.i_node  = ["pt", "eta", "phi", "energy"]
md.i_graph = ["met", "phi"]
md.device  = "cuda:0"

k = ("MRK-1", "adam", {"lr" : 1e-4}),
opti = OptimizerConfig()
opti.Optimizer = k[0][1]
for t in k[0][2]: setattr(opti, t, k[0][2][t])

ana = Analysis()
ana.kFold = [1]
ana.AddSamples(root1, "tmp")
ana.AddEvent(ev, "tmp")
ana.AddGraph(gr, "tmp")
ana.AddModel(md, opti, "run-name")
ana.GraphCache = "./ProjectName/"
ana.Targets = ["top_edge"]

ana.Start()
