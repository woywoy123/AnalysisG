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
















class GraphPainter_OLD(SharedMethods):

    def __init__(self, networkx_graph):
        self.G = networkx_graph.G
        self.Filename = ""
        self.PLT = plt
        self.PLT.figure()
        self.Title = "Graph"
        self.DrawAttribute = "weight"   
    
    def DrawAndSave(self, dir):
        SharedMethods.__init__(self)       

        # package requirement: https://bobswift.atlassian.net/wiki/spaces/GVIZ/pages/20971549/How+to+install+Graphviz+software
        pos = nx.nx_agraph.graphviz_layout(self.G)
        for i in self.G.edges:
            for j in self.G[i[0]][i[1]]:
                if isinstance(self.G[i[0]][i[1]][j], list):
                    self.G[i[0]][i[1]][j] = round(self.G[i[0]][i[1]][j][0], 2)
                else:
                    self.G[i[0]][i[1]][j] = round(self.G[i[0]][i[1]][j], 2)


        nx.draw(self.G, pos = pos)
        labels = nx.get_edge_attributes(self.G, self.DrawAttribute)
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels= labels, label_pos = 0.75, font_color="red")
        self.SaveFigure(dir)
        self.PLT.close()
