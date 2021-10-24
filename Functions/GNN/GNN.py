from Functions.GNN.Metrics import EvaluationMetrics
from Functions.Tools.Alerting import Notification
import torch 
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from Functions.GNN.GNN_Models import EdgeConv

class Optimizer(Notification):

    def __init__(self, DataLoaderObject):
        self.Optimizer = ""
        self.Model = ""
        self.DataLoader = DataLoaderObject.DataLoader
        self.Epochs = 50
        self.LearningRate = 0.01
        self.WeightDecay = 1e-8
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

        for epoch in range(self.Epochs):
            for n in self.DataLoader:
                self.DataLoader[n].shuffle = False
                for data in self.DataLoader[n]:
                    data.to(self.Device)
                    self.data = data
                    self.Learning()
            self.Notify("EPOCHLOOP::Training::EPOCH " + str(epoch+1) + "/" + str(self.Epochs) + " -> Current Loss: " + str(float(self.L)))


    def Prediction(self, in_channel, output_dim):
        _, y_p = self.Model(in_channel.x, in_channel.edge_index).max(dim = output_dim)
        return y_p

    def DefineEdgeConv(self, in_channels, out_channels):
        self.Model = EdgeConv(in_channels, out_channels)
        self.Model.to(self.Device)
        self.Model.train()
        self.Optimizer = torch.optim.Adam(self.Model.parameters(), lr = self.LearningRate, weight_decay = self.WeightDecay)
   
    def AssignPredictionToEvents(self, EventGeneratorContainer, SystematicTree):
        def AppendToList(EventParticles, out):
            for k in EventParticles:
                if isinstance(k, str):
                    continue
                else:
                    out.append(k)
            return out
        
        for i in self.LoaderObject.EventData:
            for l in self.LoaderObject.EventData[i]:
                eN = l["EventIndex"]
                Event = EventGeneratorContainer.Events[eN][SystematicTree]
                SummedParticles = []
                SummedParticles = AppendToList(Event.TruthTops, SummedParticles)
                SummedParticles = AppendToList(Event.TruthChildren, SummedParticles)
                SummedParticles = AppendToList(Event.TruthChildren_init, SummedParticles)
                SummedParticles = AppendToList(Event.TruthJets, SummedParticles)
                SummedParticles = AppendToList(Event.Muons, SummedParticles)
                SummedParticles = AppendToList(Event.Electrons, SummedParticles)
                SummedParticles = AppendToList(Event.Jets, SummedParticles)
                SummedParticles = AppendToList(Event.RCJets, SummedParticles)

                pred = self.Prediction(l, 1)
                
                for p, p_i, po in zip(l["ParticleType"], l["ParticleIndex"], pred):
                    for pe in SummedParticles:
                        if pe.Type == p and pe.Index == p_i:
                            setattr(pe, "ModelPredicts", po.item())

