import torch 
from torch.nn import CosineSimilarity
from torch.nn import Sequential as Seq, Linear, ReLU, Tanh, Sigmoid, Softmax
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
        self._out = 1
        self._mass_mlp = Seq(
                Linear(1, 1024),
                ReLU(),
                Linear(1024, 1024),
                ReLU(),
                Linear(1024, self._out)
        )

        self._edge = Seq(
                Linear(self._out, 1024), 
                ReLU(),
                Linear(1024, 1024), 
                ReLU(), 
                Linear(1024, 2)
        )
        
        self._confidence = 0

        self.O_Index = None
        self.L_Index = "CEL"
        self.C_Index = True
        self._pdf = None

    def forward(self, edge_index, N_pT, N_eta, N_phi, N_energy, E_T_Index): #Pmu_cart):
        
        Pmu_cart = TensorToPxPyPzE(torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1))
        self.O_Index = self.propagate(edge_index, Pmu = Pmu_cart, T = E_T_Index)
        src = self.O_Index[edge_index[0]]
        dst = self.O_Index[edge_index[1]]
        self.O_Index = dst - src

    def message(self, index, Pmu_i, Pmu_j):
        return Pmu_j
    
    def aggregate(self, message, index, Pmu, T):
        torch.set_printoptions(profile="full", linewidth = 200, sci_mode = False)
        OwnMass = MassFromPxPyPzE(Pmu)/1000
        IncomingMass = MassFromPxPyPzE(message) / 1000

        # Count number of unique edges incident on nodes
        cons, n_cons = index.unique(sorted = False, return_counts = True)
        lim = n_cons.max()
            
        # Construct the adjacency matrix based on maximal incoming connections
        empty = torch.zeros((Pmu.shape[0], lim), device = Pmu.device, dtype = torch.long)
        msk = n_cons[cons] != lim
        empty[msk, n_cons[msk]] = 1
        empty = 1 - empty.cumsum(dim = -1)
        msk = empty.view(-1) != 0
        ranger = torch.arange(lim, device = Pmu.device)

        # Derive a bias pdf
        link = empty.clone().to(dtype = torch.float)
        link[empty != 0] += T.view(-1)#*(index.shape[0] - torch.arange(index.shape[0], device = Pmu.device))
        
        # === Continue here with a method to bias the pdf
        src = lim*empty - ranger.to(dtype = link.dtype)*empty
        indices = torch.multinomial(link, lim)
        #link = src.scatter(1, indices, src)*empty
        ## === Continue here with a method to bias the pdf

        # Sample this distribution and perform cummulative summation of the four vector and calculate the invariant mass but removing non connected edges
        link_map, indices = link.sort(dim = 1, descending = True)
        lm_msk = link_map.view(-1) > 0 
        l_msk = link.view(-1) > 0

        IndxMap = empty.clone()
        IndxMap[empty != 0] = torch.arange(index.shape[0], device = Pmu.device).to(dtype = empty.dtype)
        
        # Collect relevant indices
        IndxMap = torch.gather(IndxMap, 1, indices.to(dtype = torch.int64)).to(dtype = torch.long)
        pmu_j = message*l_msk[msk].view(-1, 1)
        pmu_j = pmu_j[IndxMap].cumsum(dim = 1).view(-1, 4)[lm_msk]
        Mass = MassFromPxPyPzE(pmu_j) / 1000
        
        order = IndxMap.view(-1)[msk].sort()
        #Mass[order[1]] = Mass[order[0]]
         
        # Predict whether the masses derived from the cummulative summation of the four vectors are consistent
        sel = self._mass_mlp(T.view(-1, 1))
        pack = sel #self._mass_mlp(Mass)#*self._mass_mlp(OwnMass[index])
        selc = self._edge(pack)
        SelC = selc.max(1)[1] 

        ### Record the mass and bias mlp output
        Node_Bias = torch.zeros((Pmu.shape[0], self._out), device = Pmu.device)
        Node_Bias.index_add_(0, index, sel)
        return Node_Bias



        return Aggr_mlp





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




















