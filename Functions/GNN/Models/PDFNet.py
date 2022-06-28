import torch 
from torch.nn import CosineSimilarity
from torch.nn import Sequential as Seq, Linear, ReLU, Tanh, Sigmoid
import torch.nn.functional as F
from LorentzVector import *
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import *

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


class GraphNeuralNetwork_MassTagger(MessagePassing):

    def __init__(self):
        super(GraphNeuralNetwork_MassTagger, self).__init__(aggr = None, flow = "target_to_source")
        self._mass_mlp = Seq(
                Linear(5, 256),
                Linear(256, 256), 
                Linear(256, 256)
        )
        
        self._edge_mass = Seq(
                Linear(256, 256), 
                ReLU(),
                Linear(256, 128), 
                ReLU(), 
                Linear(128, 2)
        )
        
        self._edge_mlp = Seq(
                Linear(4, 4), 
                Linear(4, 2)
        )


    def forward(self, edge_index, Pmu_cart):
        edge_index = add_remaining_self_loops(edge_index)[0]
        return self.propagate(edge_index, Pmu = Pmu_cart)

    def message(self, index, Pmu_i, Pmu_j):
        Pmu = Pmu_i + Pmu_j
        Mass = MassFromPxPyPzE(Pmu)
        return self._mass_mlp(torch.cat([Pmu, Mass], dim = 1)), Pmu_j
    
    def aggregate(self, message, index, Pmu):
        e_mlp = message[0]
        Pmu_inc = message[1]
        
        mass_mlp = self._edge_mass(e_mlp)
        mass_bool = mass_mlp.max(1)[1]
        for i in range(Pmu.shape[0]):
            swi = mass_bool[i == index].view(-1, 1)
            P_inc = torch.cumsum(Pmu_inc[i == index]*swi, dim = 0)
            if i == 0:
                Psum = P_inc
                continue
            Psum = torch.cat([Psum, P_inc], dim = 0)

        mass = MassFromPxPyPzE(Psum)
        Masses = mass[mass != 0].unique() 
        
        mass = self._mass_mlp(torch.cat([Psum, mass], dim = 1))
        mass = self._edge_mass(mass)
        
        return mass_mlp, Masses #self._edge_mlp(torch.cat([mass, mass_mlp], dim = 1)), Masses


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

        # Graph Neural Network which uses particle masses to tag the topology
        self._GNN_MassTag = GraphNeuralNetwork_MassTagger()
        
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
        self.C_Index = True

        self._MassPDF = None

    def forward(self, N_eta, N_energy, N_pT, N_phi, edge_index):
        
        if self._MassPDF == None:
            self._MassPDF = torch.zeros(self._nodes, device = edge_index.device)

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
        Pmu_PP = torch.cat([self.O_pT, self.O_eta, self.O_phi, self.O_energy], dim = 1)
        Pmu_DP = torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1)

        Pmu_PC = TensorToPxPyPzE(Pmu_PP)
        Pmu_DC = TensorToPxPyPzE(Pmu_DP)
    
        delta_enc = self._deltaEnc((Pmu_DC-Pmu_PC)/(torch.abs(Pmu_DC)))
        
        # Graph Neural Network - Mass Tagging 
        TopoMass, TaggedMass = self._GNN_MassTag(edge_index, Pmu_DC)
        self.O_Index = TopoMass
        
        if TaggedMass.shape[0]:
            
            print(TaggedMass/1000)
            #TMP = torch.zeros(self._nodes - TaggedMass.shape[0], device = edge_index.device)
            #self._MassPDF = torch.cat([TMP, TaggedMass, self._MassPDF], dim = 0).sort()[0].view(-1, 2).sum(dim = 1)
            
            #print(self._MassPDF)

        return self.O_eta, self.O_energy, self.O_phi, self.O_pT, self.O_Index




















