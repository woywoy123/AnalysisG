from AnalysisTopGNN.Notification import RandomSamplers
import random

class RandomSamplers(RandomSamplers):

    def __init__(self):
        pass
    
    def RandomEvents(self, Events, nEvents):
        Indx = []
        if isinstance(Events, dict):
            Indx += list(Events.values())
        else:
            Indx += Events
        
        random.shuffle(Indx)
        if len(Events) < nEvents:
            self.NotEnoughEvents(Events, nEvents)
            return Indx    
        return Indx[0:nEvents]
