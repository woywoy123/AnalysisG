from .Notification import Notification

class SampleTracer(Notification):

    def __init__(self):
        pass

    def DifferentClassName(self, Name1, Name2):
        self.Warning("Adding two different Event classes: " + Name1 + " and " + Name2)
