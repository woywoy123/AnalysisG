import networkx as nx
from torch_geometric.utils.convert import from_networkx

class CreateEventGraph:

    def __init__(self, Event):
        self.G = nx.Graph()
        self.Event = Event
        self.EdgeAttributes = []
        self.NodeAttributes = []

    def CreateParticleNodes(self):
        self.Nodes = []
        for i in self.Event:
            self.Nodes.append(i.Index)

        self.G.add_nodes_from(self.Nodes)

    def CreateParticlesEdgesAll(self, ExcludeSelfLoops = True):
        self.Edges = []
        for i in self.Event:
            i_e = i.Index 
            for j in self.Event:
                j_e = j.Index

                if ExcludeSelfLoops and i_e == j_e:
                    continue
                self.Edges.append((i_e, j_e))
        self.G.add_edges_from(self.Edges)

    def CreateDefaultEdgeWeights(self, EdgeWeight = 1):
        
        for i in self.Edges:
            i_e = i[0]
            i_j = i[1]

            self.G[i_e][i_j]["weight"] = EdgeWeight

    
    def CalculateNodeDifference(self, NodeAttr = True, ExcludeSelfLoops = True):
        
        if NodeAttr == True:
            pass
        else:
            self.NodeAttributes = NodeAttr
        
        print(self.NodeAttributes)
        for i in range(len(self.NodeAttributes)):
            attr = self.NodeAttributes[i]
            d_label = str("d_" + attr)
            for n in self.Event:
                n_i = n.Index
                
                for k in self.Event:
                    k_i = k.Index
                    
                    if k_i == n_i and ExcludeSelfLoops:
                        continue

                    k_a = getattr(n, attr)
                    n_a = getattr(n, attr)
                    
                    dif = k_a - n_a

                    self.G[n_i][k_i][d_label] = dif
            self.EdgeAttributes.append(d_label)


    def ConvertToData(self):
        # Need to figure out why I cant edit input attributes as described here: 
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.from_networkx
        return from_networkx(self.G) 
