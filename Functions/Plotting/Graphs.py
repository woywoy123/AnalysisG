from Functions.Plotting.Histograms import SharedMethods
import matplotlib.pyplot as plt
import networkx as nx

class GraphPainter(SharedMethods):

    def __init__(self, networkx_graph):
        self.G = networkx_graph.G
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
                self.G[i[0]][i[1]][j] = round(self.G[i[0]][i[1]][j], 2)

        nx.draw(self.G, pos = pos)
        labels = nx.get_edge_attributes(self.G, self.DrawAttribute)
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels= labels, label_pos = 0.75, font_color="red")
        self.SaveFigure(dir)
