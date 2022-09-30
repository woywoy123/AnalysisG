from AnalysisTopGNN.Generators import ModelImporter
from AnalysisTopGNN.Tools import Notification
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse
import torch 
import LorentzVector as LV

class Reconstructor(ModelImporter, Notification):
    def __init__(self, Model = None):
        Notification.__init__(self)
        self.VerboseLevel = 0
        self.Caller = "Reconstructor"
        self.TruthMode = False
        self._init = False
        self.Model = Model
    
    def __call__(self, sample):
        self.Sample = sample
        if self.Model == None:
            self.TruthMode = True 

        if self._init == False and self.TruthMode == False:
            self.Device = "cuda" if sample.is_cuda else "cpu"
            self.InitializeModel()
        
        if self.TruthMode:
            self._Results = self.Sample
        else:
            self.Model.eval()
            self.MakePrediction(self.Sample)
            self._Results = self.Output(self.ModelOutputs, self.Sample) 
            self._Results = { "O_" + i : self._Results[i][0] for i in self._Results}
        return self


    def __SummingNodes(self, msk, edge_index, pt, eta, phi, e):
        
        device = edge_index.device
        Pmu = torch.cat([self.Sample[pt], self.Sample[eta], self.Sample[phi], self.Sample[e]], dim = 1)
        Pmu = LV.TensorToPxPyPzE(Pmu)        
        
        # Get the prediction of the sample and extract from the topology the number of unique classes
        edge_index_r = edge_index[0][msk == True]
        edge_index_s = edge_index[1][msk == True]

        # Weird floating point inaccuracy. When using Unique, the values seem to change slightly
        Pmu = Pmu.to(dtype = torch.long)
        Pmu_n = torch.zeros(Pmu.shape, device = Pmu.device, dtype = torch.long)
        Pmu_n.index_add_(0, edge_index_r, Pmu[edge_index_s])

        #Make sure to find self loops - Avoid double counting 
        excluded_self = edge_index[1] == edge_index[0]
        excluded_self[msk == True] = False
        Pmu_n[edge_index[0][excluded_self]] += Pmu[edge_index[1][excluded_self]]
   
        Pmu_n = (Pmu_n/1000).to(dtype = torch.long)
        Pmu_n = torch.unique(Pmu_n, dim = 0)

        return LV.MassFromPxPyPzE(Pmu_n).view(-1)

    def MassFromNodeFeature(self, TargetFeature, pt = "N_pT", eta = "N_eta", phi = "N_phi", e = "N_energy"):
        if self.TruthMode:
            TargetFeature = "N_T_" + TargetFeature
        else:
            TargetFeature = "O_" + TargetFeature
        
        # Get the prediction of the sample 
        pred = self._Results[TargetFeature].to(dtype = int).view(-1)
        
        # Filter out the nodes which are not equally valued and apply masking
        edge_index = self.Sample.edge_index
        mask = pred[edge_index[0]] == pred[edge_index[1]]
        return self.__SummingNodes(mask, edge_index, pt, eta, phi, e)
 
    def MassFromEdgeFeature(self, TargetFeature, pt = "N_pT", eta = "N_eta", phi = "N_phi", e = "N_energy"):
        if self.TruthMode:
            TargetFeature = "E_T_" + TargetFeature
        else:
            TargetFeature = "O_" + TargetFeature
        
        pred = self._Results[TargetFeature].to(dtype = int).view(-1)
        mask = pred == 1
        return self.__SummingNodes(mask, self.Sample.edge_index, pt, eta, phi, e)


      

