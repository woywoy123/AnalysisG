from AnalysisG.Generators.Interfaces import _Interface
from AnalysisG.Notification.nTupler import _nTupler
from AnalysisG.Tracer import SampleTracer
import h5py

class nTupler(_Interface):

    def __init__(self, inpt = None):
        self._DumpAsHists = {}
        self._DumpAsList = {}
        self._DumpAsPicke = {}
        self.Caller = "nTupler"
        self.Verbose = 3
        self.Files = []
        if inpt is None: return
        self.InputSelection(inpt)

    def __scrape_key__(self, inpt):
        pass



    def __reader__(self):
        for i in self.Files:
            print(i)

