from AnalysisG.Generators import EventGenerator
from AnalysisG.Events import BSM4Tops

root1 = "./samples/dilepton/*"

x = BSM4Tops()
evg = EventGenerator()
evg.Files = root1
evg.ImportEvent(x)
