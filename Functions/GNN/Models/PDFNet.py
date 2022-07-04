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
        self._out = 256
        self._mass_mlp = Seq(
                Linear(1, 1024),
                ReLU(),
                Linear(1024, 1024),
                ReLU(),
                Linear(1024, self._out)
        )

        self._edge = Seq(
                Linear(256+1, 1024), 
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
        self.O_Index = self._edge(dst-src)

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

        ## Apply an MLP on the cummulative sum of the four vectors invariant mass - Creates an inital bias
        #IncomingSumNode = empty.clone()
        #IncomingSumNode[empty != 0] = torch.arange(index.shape[0], device = Pmu.device).to(dtype = empty.dtype)
        #MassA = MassFromPxPyPzE(message[IncomingSumNode].cumsum(dim = 1).view(-1, 4)[msk])/1000
        #SumNodeMass = self._edge_bias(torch.cat([MassA, index.view(-1, 1)], dim = 1))
        #ConnectionTrigger = SumNodeMass.max(1)[1] 
        #
        ## Use the connection trigger to zero out incoming edges and recomputed the cummulative four vector
        #Blind = message.clone()
        #Blind[ConnectionTrigger > self._confidence] = message[ConnectionTrigger > self._confidence]*0
        #MassB = MassFromPxPyPzE(Blind[IncomingSumNode].cumsum(dim = 1).view(-1, 4)[msk])/1000
        #MassBias = self._edge_bias(torch.cat([MassB, index.view(-1, 1)], dim = 1))
        
        # Derive a bias pdf
        link = empty.clone().to(dtype = torch.float)
        #n_bias = MassBias.max(1)[1] 
        link[empty != 0] = T.view(-1) 
        #link[empty != 0] += n_bias.to(dtype = torch.float)
        # === Continue here with a method to bias the pdf
        indices = torch.multinomial(link, lim) 
        # === Continue here with a method to bias the pdf

        # Sample this distribution and perform cummulative summation of the four vector and calculate the invariant mass but removing non connected edges
        link_map, indices = link.sort(descending = True)
        IndxMap = empty.clone()
        IndxMap[empty != 0] = torch.arange(index.shape[0], device = Pmu.device).to(dtype = empty.dtype)
        
        # Collect relevant indices
        IndxMap = torch.gather(IndxMap, 1, indices.to(dtype = torch.int64)).to(dtype = torch.long)

        l_msk = link.view(-1) > 0
        lm_msk = link_map.view(-1) > 0 
        pmu_j = message*l_msk[msk].view(-1, 1)
        pmu_j = pmu_j[IndxMap].cumsum(dim = 1).view(-1, 4)[lm_msk]
        Mass = MassFromPxPyPzE(pmu_j) / 1000
        
        # Predict whether the masses derived from the cummulative summation of the four vectors are consistent
        sel = self._mass_mlp(Mass)
        selc = self._edge(torch.cat([sel, Mass], dim = 1))
        print(Mass[selc.max(1)[1] == 1])

        ### Record the mass and bias mlp output
        Node_Bias = torch.zeros((Pmu.shape[0], self._out + 1), device = Pmu.device)
        #Node_Mass = torch.zeros((Pmu.shape[0], self._out), device = Pmu.device)
        #Node_pdf = torch.zeros((Pmu.shape[0], self._out), device = Pmu.device)
        
        index = index[l_msk[msk]]
        
        #print(Mass, index) 
        Node_Bias.index_add_(0, index, torch.cat([sel, Mass], dim =1))
        #Node_Bias = torch.cat([Node_Bias, Pmu], dim =1)

        #Node_Mass[index] += sel
        #Node_pdf[index] += SumNodeMass
            
        # Append the encoded node's features 
        #EncodedNode = self._EncodeNode(torch.cat([OwnMass, Node_Mass], dim = 1))
        return Node_Bias




        print("____")
        print(Pmu)
        print(indices)
        print(empty)
        
        
        print(Mass)


    
        print("____")

        print()

       





        #print(torch.cat([indices_flat.view(-1, 1), index.view(-1, 1)], dim = 1))
        #
        #print(cons)




        exit()
        trans = Pmu[indices].cumsum(dim = 1).view(-1, 4)
        Mass = MassFromPxPyPzE(trans[empty.view(-1) != 0])/1000
        
        print(Mass.unique())

         


        
        print(index)



        print(cons, n_cons)

        print(indices)
        print(indices.view(-1, 1)[empty.view(-1, 1) != 0])






















        # Calculate the mass and let the NN assign a score
        #mass = MassFromPxPyPzE(message)
        #mass_sc = self._mass_mlp(mass)        
        
        # Use Boolean score to populate the PDF 
        #selc = mass_sc.max(1)[1].view(Pmu.shape[0], Pmu.shape[0])
        #self._pdf += selc
        
        # Derive the order of summing the edges based on the PDF and sort them 












        #indices = torch.multinomial(self._pdf, Pmu.shape[0]) 
        ##first_indices = torch.arange(indices.shape[0])[:None]
        #
        #print(index) 

        #print(indices)

        #mlp = self._mass_mlp(self._pdf.view(-1, 1).to(dtype = torch.float)) 
        #NodeState = Pmu.clone().zero_()

        #print(indices)
        #print(NodeState)
        #print(mlp) 









        #Pmu_mes = message.view(-1, Pmu.shape[0], 4)
        ##mass_sc = mass_sc.view(-1, Pmu.shape[0], 2)

        #Pmu_mes = Pmu_mes[first_indices, indices]
        ##mass_sc = mass_sc[first_indices, indices]
        #
        ## Calculate the cummulative sum of the incoming edges per node
        #Aggr = Pmu_mes.cumsum(dim = 1).view(-1, 4)
        ##mass_sc = mass_sc.view(-1, 2)

        #Aggr_Mass = MassFromPxPyPzE(Aggr)/1000 

        #Aggr_mlp = self._mass_mlp(Aggr_Mass)
        #selc = Aggr_mlp.max(1)[1].to(dtype = self._pdf.dtype)

        ## Update the PDF based on the selected edges
        ##self._pdf.scatter_add_(1, indices, selc.view(-1, 12))
        #
        ##self._pdf = self._SM(self._pdf)
        ##print(self._pdf)
        #
        #print(index)
        #print(indices.view(-1))



        #Aggr_mlp = Aggr_mlp[indices].view(-1, 2)
        
        #print(Aggr_mlp)


        #mass_sc = mass_sc[indices].view(-1, 2)
        #mlp = torch.cat([mass_sc, Aggr_mlp], dim = 1)
        return Aggr_mlp





        #Pmu_j = Pmu_j.view(-1, Pmu.shape[0], 4)
        #scored = torch.cat([Pmu_j, score], dim = 2) 
        #
        #sort = scored[:, :, -1].sort(descending = True)[1]
        #
        #print(sort)


        #first_indices = torch.arange(scored.shape[0])[:, None]
        #scored = scored[first_indices, sort].cumsum(dim = 1).view(-1, 5)[:, :-1]
        #
        #Mass = MassFromPxPyPzE(scored/1000)
        #Mass_score = self._mass_mlp(Mass)
        #
        #print(Mass.unique())
        #return inc_mlp+Mass_score





        #scores, indices = score.sort(dim = 1, descending = True)
        #Pmu_j = Pmu_j.view(-1, Pmu.shape[0], 4)
        #print(Pmu_j)
        ##print(scores)
        #indices = indices.view(-1)
        #
        #print(indices)
        #print(Pmu[indices])




        #print(self._SM(inc_mlp))
        
        #print(rnd)



        #Pmu_inc = message.view(-1, n_nodes, 4)
        #EdgeWeight = 




        #print(Pmu_inc)
        #


        #Pmu_inc = Pmu_inc.cumsum(dim = 1)
        ##Pmu_part = Pmu_inc + Pmu_own 
        #M_part = MassFromPxPyPzE(Pmu_inc.view(-1, 4)/1000)
        #
        #print(M_part)
        exit()
        


        score = self._edge_mass(self._mass_mlp(M_part))
        check = score.max(1)[1]
        #
        #Node_score = torch.zeros((Pmu.shape[0], 2), device = Pmu.device)
        #Node_score.index_add_(0, index, score)
        
        print(score)
        print(check)
        return score
        #print(message.view(-1, n_nodes-1, 5))



        #print(message, index, Pmu)


        exit()







    #def message(self, index, Pmu_i, Pmu_j):
    #    Pmu = Pmu_i + Pmu_j
    #    Mass = MassFromPxPyPzE(Pmu)
    #    return self._mass_mlp(torch.cat([Pmu, Mass], dim = 1)), Pmu_j
    #
    #def aggregate(self, message, index, Pmu):
    #    e_mlp = message[0]
    #    Pmu_inc = message[1]
    #    
    #    mass_mlp = self._edge_mass(e_mlp)
    #    mass_bool = mass_mlp.max(1)[1]
    #    for i in range(Pmu.shape[0]):
    #        swi = mass_bool[i == index].view(-1, 1)
    #        P_inc = torch.cumsum(Pmu_inc[i == index]*swi, dim = 0)
    #        if i == 0:
    #            Psum = P_inc
    #            continue
    #        Psum = torch.cat([Psum, P_inc], dim = 0)

    #    mass = MassFromPxPyPzE(Psum)
    #    Masses = mass[mass != 0].unique() 
    #    
    #    mass = self._mass_mlp(torch.cat([Psum, mass], dim = 1))
    #    mass = self._edge_mass(mass)
    #    
    #    return mass_mlp, Masses #self._edge_mlp(torch.cat([mass, mass_mlp], dim = 1)), Masses


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




















