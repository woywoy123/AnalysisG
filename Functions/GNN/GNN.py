import torch 
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

class EdgeConv(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr = "mean") 
        self.mlp = Seq(Linear(2 * in_channels, out_channels), ReLU(), Linear(out_channels, out_channels))
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x = x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim = 1)
        return self.mlp(tmp)

    def update(self, aggr_out):
        F.normalize(aggr_out)
        return aggr_out



class Optimizer:

    def __init__(self):
        self.Optimizer = ""
        self.Model = ""
        self.DataLoader = ""
        self.Epochs = 10
        self.Device = torch.device("cpu")
        self.LearningRate = 0.01
        self.WeightDecay = 1e-4
    
    def Repeat(self):

        self.Model.train()
        self.Optimizer.zero_grad()
        Loss = torch.nn.CrossEntropyLoss()

        edge_atr = []
        for i in self.data.d_Signal:
            edge_atr.append([float(i)])
        
        node_f = []
        for i in self.data.Signal:
            node_f.append([int(i)])
        
        node_f = torch.tensor(node_f, dtype= torch.float)
        edge_atr = torch.tensor(edge_atr, dtype = torch.float)
        

        y = torch.tensor(self.data.Signal.T[0], dtype = torch.long)
        
        self.data.mask = torch.tensor([1, 1, 1, 1], dtype = torch.bool)

        x = self.Model(self.data.Signal, self.data.edge_index) 

        self.L = Loss(x[self.data.mask], y[self.data.mask])
        self.L.backward()
        self.Optimizer.step()


        _, y_p = self.Model(self.data.Signal, self.data.edge_index).max(dim = 1)
        
        print(y_p, y)



    def EpochLoop(self):

        for epoch in range(self.Epochs):
            for data in self.DataLoader:
                print(data)
                self.data = data#.to_data_list()[0]
                self.Repeat()
                print(self.L)
                #break


    def DefineEdgeConv(self, in_channels, out_channels):
        self.Model = EdgeConv(in_channels, out_channels)
        self.Model.train()
        self.Optimizer = torch.optim.Adam(self.Model.parameters(), lr = self.LearningRate, weight_decay = self.WeightDecay)
        
