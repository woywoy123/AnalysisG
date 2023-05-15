from AnalysisG import Analysis
from AnalysisG.IO import UnpickleObject
from AnalysisG.Events import Event
from SingleLeptonic import SingleLepton 
import PlottingCode.SingleLepton as PSL

import os 
smpl = os.environ["Samples"]

if "Dilepton" in smpl: smpl = "/".join(smpl.split("/")[:-2]) + "/"

singlelep = smpl + "SingleLepton/ttH_tttt_m1000/" #DAOD_TOPQ1.21955717._000001.root"
dilepton  = smpl + "Dilepton/ttH_tttt_m1000/DAOD_TOPQ1.21955717._000001.root"


Ana = Analysis()
Ana.ProjectName = "SingleLepton"
Ana.Event = Event
Ana.EventCache = True
Ana.chnk = 1000
Ana.InputSample("ttH", singlelep)
Ana.AddSelection("SingleLepton", SingleLepton)
Ana.MergeSelection("SingleLepton")
Ana.Launch


# Debugging Section 
#st = SingleLepton()
#for i in Ana:
#    st(i)

i = "SingleLepton"

studies = {
    "SingleLepton" : PSL.SingleLepton
}
x = UnpickleObject(Ana.ProjectName + "/Selections/Merged/" + i + ".pkl")
studies[i](x)
