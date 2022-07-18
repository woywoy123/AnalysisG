import torch 
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid, Tanh
from LorentzVector import *
from torch_geometric.utils import *

class BasicEdgeConvolutionBaseLine(MessagePassing):

    def __init__(self):
        super().__init__(aggr= "max")
        
        self._Hidden = [2, 256, 256, 256, 2]
        LinDict = []
        for i in range(len(self._Hidden)-1):
            x1 = self._Hidden[i]
            x2 = self._Hidden[i+1]
            LinDict += [Linear(x1, x2), Linear(x2, x2)]
        self._edge = Seq(*LinDict)
    
    def forward(self, edge_index, Pmu, Mass):
        self.propagate(edge_index, Pmu = Pmu, Mass = Mass)
        return self.O_Edge

    def message(self, Pmu_i, Pmu_j, Mass_i, Mass_j):
        dR = TensorDeltaR(Pmu_i, Pmu_j)
        tmp = torch.cat([dR, Mass_i - Mass_j], dim = 1)
        self.O_Edge = self._edge(tmp)
        return self.O_Edge

class BasicMessageBaseLine(MessagePassing):
    def __init__(self):
        super().__init__(aggr= "max")
        
        self._Hidden = [6, 256, 256, 256, 2]
        LinDict = []
        for i in range(len(self._Hidden)-1):
            x1 = self._Hidden[i]
            x2 = self._Hidden[i+1]
            LinDict += [Linear(x1, x2), Linear(x2, x2)]
        
        self._mlp = Seq(*LinDict)

    def forward(self, edge_index, Pmu, Mass):
        self.O_FromRes = self.propagate(edge_index, Pmu = Pmu, Mass = Mass)
        return self.O_FromRes

    def message(self, Pmu_i, Pmu_j, Mass_i, Mass_j):
        dR = TensorDeltaR(Pmu_i, Pmu_j)
        tmp = torch.cat([Pmu_i, dR, Mass_i + Mass_j], dim = 1)
        return self._mlp(tmp)

class BasicBaseLineTruthChildren(MessagePassing):

    def __init__(self):
        super().__init__(aggr = "max")

        self._Edge = BasicEdgeConvolutionBaseLine()
        self.O_Edge = None
        self.L_Edge = "CEL"
        self.C_Edge = True

        self._Res = BasicMessageBaseLine()
        self.O_FromRes = None
        self.L_FromRes = "CEL"
        self.C_FromRes = True

        self.O_SignalSample = None
        self.L_SignalSample = "CEL"
        self.C_SignalSample = True


        self._Hidden = [13, 256, 256, 256, 2]
        LinDict = []
        for i in range(len(self._Hidden)-1):
            x1 = self._Hidden[i]
            x2 = self._Hidden[i+1]
            LinDict += [Linear(x1, x2), Linear(x2, x2)]
        self._edge = Seq(*LinDict)
    
    def forward(self, edge_index, N_eta, N_energy, N_pT, N_phi, num_nodes, i, ni):
        device = N_eta.device
        Pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1)
        Mass = MassFromPtEtaPhiE(Pmu)/1000
        
        self.O_Edge = self._Edge(edge_index, Pmu, Mass)
        self.O_FromRes = self._Res(edge_index, Pmu, Mass)
        
        # Collect some information about the lower layers that can be used for global context
        n_nodes = (num_nodes/i.shape[0]).to(dtype = torch.int)
        nodes = torch.zeros((i.shape[0], 1), device = device, dtype = torch.int)
        
        # Collect the number of nodes in batch: batch x 1
        nodes[torch.arange(i.shape[0])] = n_nodes
        
        # Collect the MLP for the resonance: batch x 1, batch x 2
        print(self.O_FromRes.max(1)[1].view(-1, n_nodes))
        NResNodes = self.O_FromRes.max(1)[1].view(-1, n_nodes).sum(dim = 1)
        NResNodes_mlp = self.O_FromRes.view(i.shape[0], -1, 2).sum(dim = 1)
        
        # Same as above just with the edges 
        NEdges = self.O_Edge.max(1)[1].view(i.shape[0], -1, n_nodes).sum(dim = 1).sum(dim = 1)
        NEdges_mlp = self.O_Edge.view(i.shape[0], -1, 2).sum(1)
        
        # Get Mass and Four Vector of Event
        Pmu_Event = Pmu.view(i.shape[0], -1, Pmu.shape[1]).sum(dim = 1)
        Mass_Event = MassFromPtEtaPhiE(Pmu_Event)/1000
        
        NResNodes =  NResNodes.view(-1, 1)
        NResNodes =  NResNodes_mlp.view(-1, 2)
        NEdges =  NEdges.view(-1, 1)
        NEdges_mlp =  NEdges_mlp.view(-1, 2) 
        Pmu_Event =  Pmu_Event.view(-1, 4) 
        Mass_Event =  Mass_Event.view(-1, 1)
        
        ev = torch.cat([nodes, NResNodes, NResNodes_mlp, NEdges, NEdges_mlp, Pmu_Event, Mass_Event], dim =1)
        self.O_SignalSample = self._edge(ev)

        return self.O_Edge, self.O_FromRes
