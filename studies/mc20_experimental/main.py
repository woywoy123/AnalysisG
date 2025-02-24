from AnalysisG import Analysis
from AnalysisG.core.io import IO
from AnalysisG.events.exp_mc20 import ExpMC20

#class MetaX(MetaLookup):
#
#    @property
#    def title(self): return mapping(self.meta.DatasetName)
#
#    @property
#    def SumOfWeights(self): return self.meta.SumOfWeights[b"sumWeights"]["total_events_weighted"]
#
#    @property
#    def GenerateData(self): return DataX(self)
#



smpls = "/home/tnom6927/Downloads/mc20_exp/AnalysisTop-BSM4-TruthMatching/dR0p30/*"
smpls = "/home/tnom6927/Downloads/mc20_exp/AnalysisTop-BSM4-TruthMatching/dR0p30/user.bdong.510177.MGPy8EG.DAOD_PHYS.e8307_s3797_r13145_p6117.mc20e_new_dR0p3_SSML_v01_output_root/*"


#ev = ExpMC20()
#ana = Analysis()
#ana.FetchMeta = True
#ana.Threads = 12
#ana.AddSamples(smpls, "dr")
#ana.AddEvent(ev, "dr")
#ana.Start()

smpl = IO(smpls)
smpl.MetaCachePath = "./meta_cache"
smpl.Trees = ["nominal_Loose"]
smpl.Leaves = ["weight_mc"]
smpl.EnablePyAMI = True
smpl.Keys

meta = smpl.MetaData()
meta = list(meta.values())[0]











