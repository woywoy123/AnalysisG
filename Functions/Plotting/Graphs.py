from Functions.Plotting.Histograms import SharedMethods
import matplotlib.pyplot as plt
import networkx as nx

class GraphPainter(SharedMethods):

    def __init__(self, networkx_graph):
        self.G = networkx_graph.G
        self.PLT = plt
        self.PLT.figure()
        self.Title = "Graph"
        SharedMethods.__init__(self) 
    
    def DrawAndSave(self, dir):
        
        # package requirement: https://bobswift.atlassian.net/wiki/spaces/GVIZ/pages/20971549/How+to+install+Graphviz+software
        pos = nx.nx_agraph.graphviz_layout(self.G)
        nx.draw(self.G, pos = pos)
        labels = nx.get_edge_attributes(self.G, "weight")
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels= labels)
        self.SaveFigure(dir)
