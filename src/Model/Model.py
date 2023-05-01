from AnalysisG.Notification import _ModelWrapper
from .LossFunctions import LossFunctions
from torch_geometric.data import Data
import torch
try:
    import PyC.Physics.Tensors.Polar as PT
    import PyC.Physics.Tensors.Cartesian as CT
except:
    pass

class ModelWrapper(_ModelWrapper):

    def __init__(self, model):
        self.Caller = "MODEL"
        self.Verbose = 3 
        self.OutputDirectory = None
        self.RunName = None
        self.Epoch = None

        self._Model = model
        self._inputs = {}
        self._outputs = {}
        
        # Mappings 
        self.o_mapping = {}
        self.i_mapping = {}

        self._train = True
        self._truth = True

        self._GetModelInputs
        self._build 

    def _scan(self, inpt, key):
        return {k : inpt[k] for k in inpt if k.startswith(key)}
    
    def _mapping(self, inpt, key):   
        return {"O" + k[3:] : k for k in inpt if k.startswith(key) and "O" + k[3:] in self._outputs}
 
    @property
    def _GetModelInputs(self):
        code = self._Model.forward.__code__
        self._inputs = {key : None for key in code.co_varnames[:code.co_argcount] if key != "self"}
        return self._inputs
  
    @property
    def _build(self):
        self._outputs = self._scan(self._Model.__dict__, "O_")
        mod = self._Model.__dict__["_modules"]
        mod = {i : mod[i] for i in mod}
        mod |= self._Model.__dict__
        c = self._scan(self._Model.__dict__, "C_")
        loss = self._scan(mod, "L_")
        self._loss = {l[2:] : LossFunctions(loss[l], c["C_" + l[2:]] if "C_" + l[2:] in c else False) for l in loss}
        
    def SampleCompatibility(self, smpl):
        self._pth = self.OutputDirectory + "/" + self.RunName 
        smpl = list(smpl.to_dict())
        self.i_mapping = {k : smpl[smpl.index(k)] for k in self._inputs if k in smpl}
      
        self.o_mapping  = self._mapping(smpl, "E_T_")
        self.o_mapping |= self._mapping(smpl, "N_T_")
        self.o_mapping |= self._mapping(smpl, "G_T_")

        if not self._iscompatible: return False
        return True 
  
    def __call__(self, data):
        self._Model(**{k : getattr(data, k) for k in self.i_mapping})        

        dc = self._Model.__dict__
        pred = {"batch" : data.batch, "edge_index" : data.edge_index}
        if self._truth: pred |= {self.o_mapping[k] : dc[k] for k in self.o_mapping}
        else: pred |= {k : dc[k] for k in self._outputs}  
        if not self._truth: return Data().from_dict(pred), None

        loss = {o[2:] : self._loss[o[2:]](pred[t], getattr(data, t)) for o, t in zip(self.o_mapping, self.o_mapping.values())}
        self._l = loss
        return Data().from_dict(pred), loss
 
    @property 
    def train(self): return self._train

    @train.setter
    def train(self, val): 
        if val: self._Model.train()
        else: self._Model.eval()
        self._train = val

    @property
    def dump(self):
        out = {"epoch" : self.Epoch, "model" : self._Model.state_dict()}
        torch.save(out, self._pth + "/" + str(self.Epoch) + "/TorchSave.pth")
   
    @property 
    def load(self):
        try: self.__init__(self._Model())
        except: pass
        lib = torch.load(self._pth + "/" + str(self.Epoch) + "/TorchSave.pth")
        self._Model.load_state_dict(lib["model"])
        self._Model.eval()

    @property
    def backward(self):
        loss = sum([self._l[x]["loss"] for x in self._l])
        if self._train: loss.backward()
        return loss

class Reconstruction:

    def __init__(self, model = None):
        self.Caller = "RECONSTRUCTION"
        self.TruthMode = False if model != None else True
        self.Model = model
        self._init = True

    def __switch(self, Sample, pre):
        shape = pre.size()
        if shape[1] > 1: pre = pre.max(1)[1].view(-1)
        else: pre = pre.view(-1)
        
        if shape[0] == Sample.edge_index.size()[1]:
            return self.MassFromEdgeFeature(Sample, pre).tolist()
        elif shape[0] == Sample.num_nodes:
            return self.MassFromNodeFeature(Sample, pre).tolist()

    def __Debatch(self, Inpt, sample):
        btch = Inpt.batch.unique()
        smples = [sample.subgraph(sample.batch == b) for b in btch]
        inpt = [Inpt.subgraph(Inpt.batch == b) for b in btch]
        return smples, inpt        

    def __call__(self, Sample):
        self.Model.SampleCompatibility(Sample) if self._init else None
        self._init = False
        self.Model._truth = self.TruthMode
        pred, truth, _ = self.Model.Prediction(Sample)

        Sample, pred = self.__Debatch(pred if self.TruthMode == False else truth, Sample)
        out = [{o[2:] : self.__switch(j, i[o[2:]]) for o in self.Model._modeloutputs} for i, j in zip(pred, Sample)]
        return out

    def __SummingNodes(self, Sample, msk, edge_index, pt, eta, phi, e):
        
        device = edge_index.device
        Pmu = torch.cat([PT.PxPyPz(Sample[pt], Sample[eta], Sample[phi]), Sample[e]], dim = -1)
        
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

        return CT.Mass(Pmu_n).view(-1)

    def MassFromNodeFeature(self, Sample, pred, pt = "N_pT", eta = "N_eta", phi = "N_phi", e = "N_energy"):
        
        # Filter out the nodes which are not equally valued and apply masking
        edge_index = Sample.edge_index
        mask = pred[edge_index[0]] == pred[edge_index[1]]
        return self.__SummingNodes(Sample, mask, edge_index, pt, eta, phi, e)
 
    def MassFromEdgeFeature(self, Sample, pred, pt = "N_pT", eta = "N_eta", phi = "N_phi", e = "N_energy"):
        mask = pred == 1
        return self.__SummingNodes(Sample, mask, Sample.edge_index, pt, eta, phi, e)

    def ClosestParticle(self, tru, pred):

        res = []
        if len(tru) == 0: return res
        if len(pred) == 0: return pred 
        p = pred.pop(0)
        max_tru, min_tru = max(tru), min(tru)
        col = True if p <= max_tru and p >= min_tru else False

        if col == False:
            if len(pred) == 0: return res
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



