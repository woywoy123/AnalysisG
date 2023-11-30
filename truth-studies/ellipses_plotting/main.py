from AnalysisG import Analysis
from AnalysisG.Events import Event 
from selection import Selection
from AnalysisG.IO import nTupler
from AnalysisG.Plotting import TH2F, TH1F
import os
import json
# import test_nusol

def launch_analysis():
    try:
        os.mkdir('Plots')
        os.mkdir('Plots/intersection_found')
        os.mkdir('Plots/intersection_not_found')
    except:
        pass
    root_dir = "/nfs/dust/atlas/user/sitnikov/ntuples_for_classifier/ttH_tttt_m1000/"
    files = {
                "1000" : root_dir + "DAOD_TOPQ1.21955717._000001.root"
    }
    ana = Analysis()
    ana.Event = Event
    ana.ProjectName = "AnalysisTruthMeVSmall"
    ana.InputSample("sample", files['1000'])
    ana.AddSelection(Selection)
    ana.Launch()



launch_analysis()