from AnalysisG.Generators import EventGenerator, GraphGenerator
from AnalysisG.Events import BSM4Tops
from AnalysisG.Graphs import TruthTops

root1 = "./samples/dilepton/*"

x = BSM4Tops()
evg = EventGenerator()
evg.Files = root1
evg.ImportEvent(x)
evg.CompileEvents()

tt = TruthTops()
egr = GraphGenerator()
egr.ImportGraph(tt)
egr.AddEvents(evg)
