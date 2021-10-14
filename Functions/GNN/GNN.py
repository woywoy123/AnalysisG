from Functions.GNN.Metrics import EvaluationMetrics
from Functions.Tools.Alerting import Notification
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



class Optimizer(Notification):

    def __init__(self, DataLoaderObject):
        self.Optimizer = ""
        self.Model = ""
        self.DataLoader = DataLoaderObject.DataLoader
        self.Epochs = 50
        self.LearningRate = 0.01
        self.WeightDecay = 1e-6
        self.Device = DataLoaderObject.Device
        self.LoaderObject = DataLoaderObject
        
        Notification.__init__(self, DataLoaderObject.Verbose)
        self.Caller = "OPTIMIZER"

    def Learning(self):

        self.Model.train()
        self.Optimizer.zero_grad()
        Loss = torch.nn.CrossEntropyLoss()

        x = self.Model(self.data.x, self.data.edge_index) 
        
        self.L = Loss(x[self.data.mask], self.data.y[self.data.mask])
        self.L.backward()
        self.Optimizer.step()

    def EpochLoop(self):
        self.Notify("EPOCHLOOP::Training")

        self.DataLoader.shuffle = True
        for epoch in range(self.Epochs):
            self.ResetWeights() 
            for data in self.DataLoader:
                data.to(self.Device)
                self.data = data
                self.Learning()
            self.Notify("EPOCHLOOP::Training::EPOCH" + str(epoch) + "/" + str(self.Epochs) + " -> Current Loss: " + str(float(self.L)))


    def Prediction(self, in_channel, output_dim):
        _, y_p = self.Model(in_channel, self.data.edge_index).max(dim = output_dim)
        return y_p

    def DefineEdgeConv(self, in_channels, out_channels):
        self.Model = EdgeConv(in_channels, out_channels)
        self.Model.to(self.Device)
        self.Model.train()
        self.Optimizer = torch.optim.Adam(self.Model.parameters(), lr = self.LearningRate, weight_decay = self.WeightDecay)
   
    def AssignPredictionToEvents(self, EventMap, SystematicTree):
        def AppendToList(EventParticles, out):
            for k in EventParticles:
                if isinstance(k, str):
                    continue
                else:
                    out.append(k)
            return out


        for i in self.LoaderObject.EventData:
            eN = i["EventIndex"]
            Event = EventMap[eN][SystematicTree]
            SummedParticles = []
            SummedParticles = AppendToList(Event.TruthTops, SummedParticles)
            SummedParticles = AppendToList(Event.TruthChildren, SummedParticles)
            SummedParticles = AppendToList(Event.TruthChildren_init, SummedParticles)
            SummedParticles = AppendToList(Event.TruthJets, SummedParticles)
            SummedParticles = AppendToList(Event.Muons, SummedParticles)
            SummedParticles = AppendToList(Event.Electrons, SummedParticles)
            SummedParticles = AppendToList(Event.Jets, SummedParticles)
            pred = self.Prediction(i.x, 1)

            for p, p_i, po in zip(i["ParticleType"], i["ParticleIndex"], pred):
                for pe in SummedParticles:
                    if pe.Type == p and pe.Index == p_i:
                        setattr(pe, "ModelPredicts", po)

