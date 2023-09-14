from AnalysisG.Notification import _FeatureAnalysis
from AnalysisG.SampleTracer import SampleTracer

class FeatureAnalysis(_FeatureAnalysis, SampleTracer):
    def __init__(self):
        SampleTracer.__init__(self)
        self.Caller = "FEATURE-ANALYSIS"
        self.GraphAttribute = {}
        self.NodeAttribute  = {}
        self.EdgeAttribute  = {}

    def TestGraphFeature(self, Event, EventGraph, Fx):
        if Fx(EventGraph(Event).Event) is not None: return
        raise AttributeError

    def TestNodeFeature(self, Event, EventGraph, Fx):
        for i in EventGraph(Event).Particles:
            if Fx(i) is not None: continue
            raise AttributeError

    def TestEdgeFeature(self, Event, EventGraph, Fx):
        ev = EventGraph(Event)
        for i in ev.Particles:
            for j in ev.Particles:
                if Fx(i, j) is not None: continue
                raise AttributeError

    def TestEvent(self, Event, EventGraph, EventIndex=None):
        if isinstance(Event, list):
            for ev in Event:
                key = " " + str(ev.index)
                if self.TestEvent(ev, EventGraph, key):
                    return True
            return False

        self.Success("!!!> Test at EventIndex:" + EventIndex)
        count = 0
        for c_name, code in self.GraphAttribute.items():
            try:
                self.TestGraphFeature(Event, EventGraph, code)
                self.PassedTest(c_name, "GRAPH")
            except AttributeError:
                self.FeatureFailure(c_name, "GRAPH", EventIndex)
                count += 1

        for c_name, code in self.NodeAttribute.items():
            try:
                self.TestNodeFeature(Event, EventGraph, code)
                self.PassedTest(c_name, "NODE")
            except AttributeError:
                self.FeatureFailure(c_name, "NODE", EventIndex)
                count += 1

        for c_name, code in self.EdgeAttribute.items():
            try:
                self.TestEdgeFeature(Event, EventGraph, code)
                self.PassedTest(c_name, "EDGE")
            except AttributeError:
                self.FeatureFailure(c_name, "EDGE", EventIndex)
                count += 1
        if count > 0: return self.TotalFailure()
        self.WhiteSpace()
        return False
