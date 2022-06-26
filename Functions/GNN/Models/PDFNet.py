import torch 
from torch.nn import CosineSimilarity
from torch.nn import Sequential as Seq, Linear, ReLU, Tanh, Sigmoid
import torch.nn.functional as F
from LorentzVector import *
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops

class BasePDFNetEncode(torch.nn.Module):
    
    def __init__(self, nodes = 256, inp = 1):
        super().__init__()
        self._mlp = Seq(Linear(inp, nodes), 
                        Linear(nodes, nodes*4))
    def forward(self, kin):
        return self._mlp(kin)


class BasePDFNetDecode(torch.nn.Module):
    
    def __init__(self, nodes = 256):
        super().__init__()
        self._mlp = Seq(Linear(nodes*4, nodes), 
                        Linear(nodes, 1))
    
    def forward(self, kin):
        return self._mlp(kin)

class BasePDFNetProbability(torch.nn.Module):
    def __init__(self, nodes = 256, inp = 1, oup = 1):
        super().__init__()
        self._mlp = Seq(Linear(inp, nodes),
                        Sigmoid(),
                        Linear(nodes, nodes),
                        Sigmoid(),  
                        Linear(nodes, oup), 
                        Sigmoid())

    def forward(self, kin):
        return self._mlp(kin)

class GraphNeuralNetwork(MessagePassing):

    def __init__(self, nodes):
        super().__init__(aggr = "add", flow = "target_to_source")
        self._edgemlp = Seq(Linear(10, nodes), 
                            ReLU(), 
                            Linear(nodes, nodes), 
                            ReLU(), 
                            Linear(nodes, 2))
    
        self._nodemlp = Seq(Linear(1, nodes), 
                            ReLU(), 
                            Linear(nodes, nodes), 
                            ReLU(), 
                            Linear(nodes, 8))

        self._aggrmlp = Seq(Linear(8+8, nodes),
                            ReLU(), 
                            Linear(nodes, nodes), 
                            ReLU(), 
                            Linear(nodes, 8))


    def forward(self, edge_index, Pmu, W_Pmu):
        #edge_index = remove_self_loops(edge_index)[0]
        aggr = self.propagate(edge_index, Pmu = Pmu, W = W_Pmu)
        return self._aggrmlp(torch.cat([aggr, Pmu, W_Pmu], dim = 1))

    def message(self, Pmu_i, Pmu_j, W_i, W_j):
        tmp = torch.pow(TensorToPtEtaPhiE(Pmu_i)[..., 1:3], 2)
        delR = torch.sqrt(torch.sum(tmp, dim = 1, keepdim = True))
        
        mass = MassFromPxPyPzE(Pmu_i + Pmu_j)
        
        # Input: 1 + 1 + 4 + 4
        return torch.cat([Pmu_j, self._edgemlp(torch.cat([delR, mass, Pmu_i - Pmu_j, W_i - W_j], dim = 1))], dim = 1)
    
    def aggregate(self, message, index, Pmu, W):
        Pmu_m = message[..., 0:4]
        message_weight = message[..., 4:]
        sel = message_weight.max(1)[1]
        index = index[ sel != 0]
        Pmu_m = Pmu_m[ sel != 0]
        
        Pmu_mass = MassFromPxPyPzE(Pmu.index_add_(0, index, Pmu_m))
        print(Pmu_mass/1000)
        return self._nodemlp(Pmu_mass)
    

class PDFNetTruthChildren(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self._nodes = 256
        
        # Autoencoding the kinematics
        self._PtEnc = BasePDFNetEncode(self._nodes)
        self._PtDec = BasePDFNetDecode(self._nodes)

        self._PhiEnc = BasePDFNetEncode(self._nodes)
        self._PhiDec = BasePDFNetDecode(self._nodes)

        self._EtaEnc = BasePDFNetEncode(self._nodes)
        self._EtaDec = BasePDFNetDecode(self._nodes)

        self._EnEnc = BasePDFNetEncode(self._nodes)
        self._EnDec = BasePDFNetDecode(self._nodes)
        
        # Learn Residual of prediction and observed kinematics
        self._deltaEnc = BasePDFNetEncode(self._nodes, 4)
        
        # Scaling factor which learns the error of the predicted kinematics 
        self._WPx = BasePDFNetProbability(self._nodes, self._nodes*4, 1)
        self._WPy = BasePDFNetProbability(self._nodes, self._nodes*4, 1)
        self._WPz = BasePDFNetProbability(self._nodes, self._nodes*4, 1)
        self._WEn = BasePDFNetProbability(self._nodes, self._nodes*4, 1)
        
        # This MLP assigns each node a weight based on the error of the autoencoder and its kinematics
        self._NodeScaling = BasePDFNetProbability(self._nodes, 8, 1)

        # Graph Neural Network - Node and Topology classifcation/prediction
        self._GNN = GraphNeuralNetwork(self._nodes)
        self._Cos = Seq(Linear(16, 4), 
                        Linear(4, 2))

        self.L_eta = "MSEL"
        self.O_eta = None
        self.C_eta = False

        self.L_phi = "MSEL"
        self.O_phi = None
        self.C_phi = False

        self.L_pT = "MSEL"
        self.O_pT = None
        self.C_pT = False

        self.L_energy = "MSEL"
        self.O_energy = None
        self.C_energy = False

        self.L_Index = "CEL"
        self.O_Index = None
        self.N_Index = True
        self.C_Index = True
        
        #self.L_expPx = "MSEL"
        #self.O_expPx = None
        #self.C_expPx = False

    def forward(self, N_eta, N_energy, N_pT, N_phi, edge_index):
        
        # Encode the kinematics 
        Pt_enc =  self._PtEnc(N_pT)
        Eta_enc = self._EtaEnc(N_eta)
        Phi_enc = self._PhiEnc(N_phi)
        E_enc =   self._EnEnc(N_energy)
        
        #Decode the kinematics
        self.O_pT =     self._PtDec(Pt_enc)  
        self.O_eta =    self._EtaDec(Eta_enc)
        self.O_phi =    self._PhiDec(Phi_enc)
        self.O_energy = self._EnDec(E_enc)
        
        # Calculate the difference between the prediction and observed
        pred = TensorToPxPyPzE(torch.cat([self.O_pT, self.O_eta, self.O_phi, self.O_energy], dim = 1))
        truth = TensorToPxPyPzE(torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1))

        delta_enc = self._deltaEnc(pred-truth)
        
        # Calculate a weight factor 
        W_Px = self._WPx(delta_enc*Eta_enc)
        W_Py = self._WPy(delta_enc*Phi_enc)
        W_Pz =  self._WPz(delta_enc*Pt_enc)
        W_En =  self._WEn(delta_enc*E_enc)
        
        Pmu = TensorToPxPyPzE(torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1))
        W_Pmu = torch.cat([W_Px, W_Py, W_Pz, W_En], dim = 1)
        WeightedNodes = self._NodeScaling(torch.cat([Pmu, W_Pmu], dim = 1))*Pmu
        
        # Add as global graph output - It uses the sum of all nodes*weight to calculate the invariant mass
        TopoGraphMass = MassFromPxPyPzE(torch.sum(WeightedNodes, dim = 0))
         
        # Predict the Topology of Graph 
        Nodes = self._GNN(edge_index, Pmu, W_Pmu)
        self.O_Index = self._Cos(torch.cat([Nodes[edge_index[0]], Nodes[edge_index[1]]], dim = 1))
        return self.O_eta, self.O_energy, self.O_phi, self.O_pT, self.O_Index




















