import sys
sys.path.append('/home/tnom6927/Dokumente/Project/Analysis/bsm4tops-gnn-analysis/AnalysisTopGNN/test/Project/CondorDump/_SharedCode/')
from AnalysisG import Analysis
from EventEx import EventEx

Ana = Analysis()
Ana.Threads = 2
Ana.Files = {'/home/tnom6927/Dokumente/Project/Analysis/bsm4tops-gnn-analysis/AnalysisTopGNN/test/samples/sample2': ['smpl2.root']}
Ana.ProjectName = 'Project'
Ana.OutputDirectory = './'
Ana.SampleMap = {'Sample1': {'/home/tnom6927/Dokumente/Project/Analysis/bsm4tops-gnn-analysis/AnalysisTopGNN/test/samples/sample1': ['smpl1.root']}, 'Sample2': {'/home/tnom6927/Dokumente/Project/Analysis/bsm4tops-gnn-analysis/AnalysisTopGNN/test/samples/sample2': ['smpl2.root']}}
Ana.EventCache = True
Ana.Event = EventEx
Ana.Launch