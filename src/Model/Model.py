from AnalysisG._cmodules.cWrapping import ModelWrapper
from AnalysisG.Notification import _ModelWrapper
from torch_geometric.data import Data
import torch

class Model(_ModelWrapper, ModelWrapper):
    def __init__(self, inpt = None):
        ModelWrapper.__init__(self, inpt)
        self.Caller = "MODEL"
        self.Verbose = 3

    def SampleCompatibility(self, smpl):
        if self._check_broken_model(): return False
        self.match_data_model_vars(smpl)
        if not self._iscompatible(smpl.to_dict()): return False
        return True

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

        diff = [abs(p - t) for t in tru]
        tru.pop(diff.index(min(diff)))
        res += self.ClosestParticle(tru, pred)
        res.append(p)
        return res

    def ParticleEfficiency(self):
        tmp = self.TruthMode
        self.TruthMode = True
        t = self.mass

        self.TruthMode = False
        p = self.mass
        output = []
        for b in range(len(t)):
            out = {}
            for f in t[b]:
                pred, truth = p[b][f], t[b][f]
                pred = self.ClosestParticle(truth, pred)
                p_l, t_l = len(pred), len(truth)
                out[f] = {
                    "%": float(p_l / (t_l if t_l != 0 else 1)) * 100,
                    "nrec": p_l,
                    "ntru": t_l,
                }
            output.append(out)
        self.TruthMode = tmp
        return output
