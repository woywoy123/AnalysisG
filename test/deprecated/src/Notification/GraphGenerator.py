from .Notification import Notification

class GraphGenerator(Notification):

    def __init__(self):
        pass
        
    def CheckSettings(self):
        
        message = False
        if self.EventGraph == None:
            message = "EventGraph not defined (obj.EventGraph). See implementations (See src/Events/EventGraphs.py)"
        
        if message:
            self.Failure("="*len(message))
            self.FailureExit(message)

        attrs = 0
        if len(list(self.EdgeAttribute)) == 0:
            self.Warning("NO EDGE FEATURES PROVIDED")
            attrs+=1
        if len(list(self.NodeAttribute)) == 0:
            self.Warning("NO NODE FEATURES PROVIDED")
            attrs+=1
        if len(list(self.GraphAttribute)) == 0:
            self.Warning("NO GRAPH FEATURES PROVIDED")
            attrs+=1
        if attrs == 3:
            message = "NO ATTRIBUTES DEFINED!"
            self.Failure("="*len(message))
            self.FailureExit("NO ATTRIBUTES DEFINED!")
        self.Success("!!Data being processed on " + self.Device)
