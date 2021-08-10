from Functions.GNN.Metrics import EvaluationMetrics
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
        self.Epochs = 100
        self.Device = torch.device("cuda")
        self.LearningRate = 0.01
        self.WeightDecay = 1e-6
    
    def Repeat(self):

        self.Model.train()
        self.Optimizer.zero_grad()
        Loss = torch.nn.CrossEntropyLoss()
        x = self.Model(self.data.x, self.data.edge_index) 

        self.L = Loss(x[self.data.mask], self.data.y[self.data.mask])
        self.L.backward()
        self.Optimizer.step()


    def EpochLoop(self):
        
        Met = EvaluationMetrics()

        #self.DataLoader.shuffle = True
        for epoch in range(self.Epochs):
            for data in self.DataLoader:
                data.to(self.Device)
                self.data = data
                self.Repeat()
            
            print(round(float(self.L), 5), Met.Accuracy(self.Prediction(data.x, 1), data.y))

    def Prediction(self, in_channel, output_dim):
        _, y_p = self.Model(in_channel, self.data.edge_index).max(dim = output_dim)
        return y_p


    def DefineEdgeConv(self, in_channels, out_channels):
        self.Model = EdgeConv(in_channels, out_channels)
        self.Model.to(self.Device)
        self.Model.train()
        self.Optimizer = torch.optim.Adam(self.Model.parameters(), lr = self.LearningRate, weight_decay = self.WeightDecay)
        
