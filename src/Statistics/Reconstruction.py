import torch 
import LorentzVector as LV

class Reconstruction:

    def __init__(self):
        self.Caller = "RECONSTRUCTION"

    def __SummingNodes(self, Sample, msk, edge_index, pt, eta, phi, e):
        
        device = edge_index.device
        Pmu = torch.cat([Sample[pt], Sample[eta], Sample[phi], Sample[e]], dim = 1)
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

    def MassFromNodeFeature(self, Sample, TargetFeature, pt = "N_pT", eta = "N_eta", phi = "N_phi", e = "N_energy"):
        
        # Get the prediction of the sample 
        pred = Sample[TargetFeature].to(dtype = int).view(-1)
        
        # Filter out the nodes which are not equally valued and apply masking
        edge_index = Sample.edge_index
        mask = pred[edge_index[0]] == pred[edge_index[1]]
        return self.__SummingNodes(mask, Sample, edge_index, pt, eta, phi, e)
 
    def MassFromEdgeFeature(self, Sample, TargetFeature, pt = "N_pT", eta = "N_eta", phi = "N_phi", e = "N_energy"):
        pred = Sample[TargetFeature].to(dtype = int).view(-1)
        mask = pred == 1
        return self.__SummingNodes(mask, Sample, Sample.edge_index, pt, eta, phi, e)

    def ClosestParticle(self, tru, pred):

        res = []
        if len(tru) == 0:
            return res
        if len(pred) == 0:
            return pred 
        p = pred.pop(0)
        max_tru, min_tru = max(tru), min(tru)
        col = True if p <= max_tru and p >= min_tru else False

        if col == False:
            if len(pred) == 0:
                return res
            return self.ClosestParticle(tru, pred)

        diff = [ abs(p - t) for t in tru ]
        tru.pop(diff.index(min(diff)))
        res += self.ClosestParticle(tru, pred)
        res.append(p)
        return res 
    
    def ParticleEfficiency(self, pred, truth, proc):
        t_, p_ = [], []
        t_ += truth
        p_ += pred 

        p = self.ClosestParticle(t_, p_)
        p_l, t_l = len(p), len(truth)

        perf = float(p_l/t_l)*100

        out = {"Prc" : proc, "%" : perf, "nrec" : p_l, "ntru" : t_l}
        
        return out


      

