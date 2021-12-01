from Functions.Plotting.Histograms import SharedMethods
import matplotlib.pyplot as plt
import networkx as nx

class GenericAttributes:

    def __init__(self):
        self.Title = ""
        self.Filename = ""
        self.EdgeKey = ""
        self.NodeKey = ""
        self.DefaultDPI = 500
        self.DefaultScaling = 8

class Graph(SharedMethods, GenericAttributes):
    
    def __init__(self, Event):
        SharedMethods.__init__(self)
        GenericAttributes.__init__(self)
        self.PLT = plt
        self.PLT.figure()
        self.Event = Event
        self.G = Event.G

    def CompileGraph(self):
        
        pos = nx.circular_layout(self.G)
        if self.EdgeKey != "":
            pass
        
        if self.Event.SelfLoop:
            self.Caller = "GRAPH-PLOTTING"
        else:
            nx.draw(self.G, pos, with_labels = True)
            nx.draw_networkx_edges(self.G, pos)


