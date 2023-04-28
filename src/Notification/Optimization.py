from .Notification import Notification 

class _Optimizer(Notification):
    
    def __init__(self):
        pass

    @property
    def _NoModel(self):
        if self.Model is not None: return False
        self.Warning("No Model was given.")
        return True

    @property
    def _NoSampleGraph(self):
        l = len(self)
        if l == 0: return self.Warning("No Sample Graphs found")
        self.Success("Found " + str(l) + " Sample Graphs") 
        return False

    @property
    def _notcompatible(self):
        return self.Failure("Model not compatible with given input graph sample.")
