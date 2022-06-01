from Functions.GNN.Optimizer import ModelImporter
from Functions.Tools.Alerting import Notification
from torch_geometric.data import Batch
from torch_geometric.utils import remove_self_loops, to_dense_adj, dense_to_sparse
import torch 
import LorentzVector as LV

class ParticleReconstructor(ModelImporter, Notification):
    def __init__(self, Model, Sample):
        Notification.__init__(self)
        self.VerboseLevel = 0
        self.Caller = "ParticleReconstructor"
        
        if Sample.is_cuda:
            self.Device = "cuda"
        else:
            self.Device = "cpu"
        self.Model = Model
        self.Sample = Sample
        self.InitializeModel()
    
    def Prediction(self):
        self.Model.eval()
        self.MakePrediction(Batch.from_data_list([self.Sample]))
        self.__Results = self.Output(self.ModelOutputs, self.Sample) 

    def MassFromFeature(self, TargetFeature, pt = "N_pt", eta = "N_eta", phi = "N_phi", e = "N_energy"):
        def To1D(inpt):
            return inpt.view(1, -1)[0]

        pt_s = To1D(self.Sample[pt])
        eta_s = To1D(self.Sample[eta])
        phi_s = To1D(self.Sample[phi])
        e_s = To1D(self.Sample[e])

        edge_index_s = self.Sample.edge_index[0]
        edge_index_r = self.Sample.edge_index[1]
        nodes = self.Sample.num_nodes
        pred = To1D(self.Sample.N_T_Index).to(dtype = int)
     
        print(pt_s)


        pt_T = To1D(self.Sample.E_T_Topology).view(-1, nodes)*pt_s
        eta_T = To1D(self.Sample.E_T_Topology).view(-1, nodes)*eta_s 
        phi_T = To1D(self.Sample.E_T_Topology).view(-1, nodes)*phi_s
        e_T = To1D(self.Sample.E_T_Topology).view(-1, nodes)*e_s
        
        print(self.Sample.E_T_Topology)



        edge_index_i = edge_index_s[pred[edge_index_r] == pred[edge_index_s]]
        edge_index_j = edge_index_r[pred[edge_index_r] == pred[edge_index_s]]
            

        edge_index = torch.cat([edge_index_i, edge_index_j], dim = 0).view(-1, len(edge_index_i))
        mat = to_dense_adj(remove_self_loops(edge_index)[0])[0]
        for i in range(1, nodes):
            mat.diagonal(-i).zero_()
        edge_index = dense_to_sparse(mat)[0] 

        pt_ = torch.sum(pt_s[edge_index], dim = 0).view(-1, 1)
        eta_ = torch.sum(eta_s[edge_index], dim = 0).view(-1, 1)
        phi_ = torch.sum(phi_s[edge_index], dim = 0).view(-1, 1)
        e_ = torch.sum(e_s[edge_index], dim = 0).view(-1, 1)

        four_vec = torch.cat([pt_, eta_, phi_, e_], dim = 1)
        print(four_vec) 
        print(pred[edge_index][0]-1)
        
        particles_ = torch.zeros((len(torch.unique(pred[edge_index][0])), 4), device = self.Model.Device)
        particles_[pred[edge_index][0]-1] += four_vec
        print(LV.MassFromPtEtaPhiE(particles_)/1000)









        exit()


        print(self.Sample)

        print(pt_s)
        print(pred)
        print("")
        print(pred[edge_index_s]) #*pred[edge_index_r])


        exit()
        if len(pred) == self.Sample.num_nodes:
            print(pt_s[edge_index_s])
            print(pt_s[edge_index_r])

            print(torch.dot(edge_index_r, edge_index_s))





            print(LV.MassFromPtEtaPhiE(torch.tensor([torch.sum(pt_s*pred), torch.sum(eta_s*pred), torch.sum(phi_s*pred), torch.sum(e_s*pred)]))/1000)
        else:





            pass 


