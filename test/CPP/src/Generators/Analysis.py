from .SelectionGenerator import SelectionGenerator 
from AnalysisG.Notification import _Analysis
from .EventGenerator import EventGenerator 
from .GraphGenerator import GraphGenerator 
from AnalysisG.Tracer import SampleTracer
from AnalysisG.Settings import Settings
from .Interfaces import _Interface

class Analysis(_Analysis, Settings, SampleTracer, _Interface):
    
    def __init__(self):
        self.Caller = "ANALYSIS"
        _Analysis.__init__(self)
        _Interface.__init__(self)
        Settings.__init__(self)
        SampleTracer.__init__(self)
  
    @property
    def __build__(self):

        self._cPWD = self.pwd
        if self.OutputDirectory is None:
            self.OutputDirectory = self._cPWD
        else:
            self.OutputDirectory = self.abs(self.OutputDirectory)
        self.OutputDirectory = self.AddTrailing(self.OutputDirectory, "/")
        self.OutputDirectory += self.ProjectName
        self.OutputDirectory += "/"
        
        if self.PurgeCache: self._WarningPurge
        self._BuildingCache 

    @property
    def __Selection__(self):
        sel = SelectionGenerator(self)
        sel.ImportSettings(self)
        for name in self.Selections: sel.AddSelection(name, self.Selections[name])
        sel.MakeSelection
    
    @property 
    def __Event__(self):
        ev = EventGenerator(self.samples)
        ev.ImportSettings(self)
        return ev.MakeEvents

    @property
    def __Graph__(self):
        gr = GraphGenerator(self)
        gr.ImportSettings(self)
        return gr.MakeGraphs
    
    @property
    def Launch(self):
        self.__build__
        for i in self.SampleMap:
            self.samples = self.SampleMap[i]
            #self.__Event__ 
            self.__Graph__


        exit()
        self.__Selection__



