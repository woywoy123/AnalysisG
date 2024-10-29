import AnalysisG
from AnalysisG.core import IO
from AnalysisG.core.plotting import *
from AnalysisG.generators import Analysis

root = "/home/tnom6927/Downloads/mc16-full/"
sets = [
    root+"user.tnommens.342284.Pythia8EvtGen.DAOD_TOPQ1.e4246_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.342284.Pythia8EvtGen.DAOD_TOPQ1.e4246_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.342284.Pythia8EvtGen.DAOD_TOPQ1.e4246_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.342285.Pythia8EvtGen.DAOD_TOPQ1.e4246_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.342285.Pythia8EvtGen.DAOD_TOPQ1.e4246_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.342285.Pythia8EvtGen.DAOD_TOPQ1.e4246_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.346344.PhPy8EG.DAOD_TOPQ1.e7148_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.346344.PhPy8EG.DAOD_TOPQ1.e7148_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.346344.PhPy8EG.DAOD_TOPQ1.e7148_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.346345.PhPy8EG.DAOD_TOPQ1.e7148_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.346345.PhPy8EG.DAOD_TOPQ1.e7148_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.346345.PhPy8EG.DAOD_TOPQ1.e7148_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.411073.PhPy8EG.DAOD_TOPQ1.e6798_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.411073.PhPy8EG.DAOD_TOPQ1.e6798_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.411074.PhPy8EG.DAOD_TOPQ1.e6798_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.411074.PhPy8EG.DAOD_TOPQ1.e6798_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.411074.PhPy8EG.DAOD_TOPQ1.e6798_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.411075.PhPy8EG.DAOD_TOPQ1.e6798_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.411075.PhPy8EG.DAOD_TOPQ1.e6798_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.411075.PhPy8EG.DAOD_TOPQ1.e6798_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.411076.PhPy8EG.DAOD_TOPQ1.e6798_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.411076.PhPy8EG.DAOD_TOPQ1.e6798_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.411076.PhPy8EG.DAOD_TOPQ1.e6798_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.411077.PhPy8EG.DAOD_TOPQ1.e6798_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.411077.PhPy8EG.DAOD_TOPQ1.e6798_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.411077.PhPy8EG.DAOD_TOPQ1.e6798_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.411078.PhPy8EG.DAOD_TOPQ1.e6798_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.411078.PhPy8EG.DAOD_TOPQ1.e6798_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.411078.PhPy8EG.DAOD_TOPQ1.e6798_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.407342.PhPy8EG.DAOD_TOPQ1.e6414_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.407342.PhPy8EG.DAOD_TOPQ1.e6414_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.407342.PhPy8EG.DAOD_TOPQ1.e6414_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.407343.PhPy8EG.DAOD_TOPQ1.e6414_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.407343.PhPy8EG.DAOD_TOPQ1.e6414_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.407343.PhPy8EG.DAOD_TOPQ1.e6414_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.407344.PhPy8EG.DAOD_TOPQ1.e6414_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.407344.PhPy8EG.DAOD_TOPQ1.e6414_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.407344.PhPy8EG.DAOD_TOPQ1.e6414_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410470.PhPy8EG.DAOD_TOPQ1.e6337_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410470.PhPy8EG.DAOD_TOPQ1.e6337_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410470.PhPy8EG.DAOD_TOPQ1.e6337_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410472.PhPy8EG.DAOD_TOPQ1.e6348_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410472.PhPy8EG.DAOD_TOPQ1.e6348_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410472.PhPy8EG.DAOD_TOPQ1.e6348_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410480.PhPy8EG.DAOD_TOPQ1.e6454_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410480.PhPy8EG.DAOD_TOPQ1.e6454_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410480.PhPy8EG.DAOD_TOPQ1.e6454_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410482.PhPy8EG.DAOD_TOPQ1.e6454_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410482.PhPy8EG.DAOD_TOPQ1.e6454_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410482.PhPy8EG.DAOD_TOPQ1.e6454_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410658.PhPy8EG.DAOD_TOPQ1.e6671_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410658.PhPy8EG.DAOD_TOPQ1.e6671_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410658.PhPy8EG.DAOD_TOPQ1.e6671_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410659.PhPy8EG.DAOD_TOPQ1.e6671_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410659.PhPy8EG.DAOD_TOPQ1.e6671_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410659.PhPy8EG.DAOD_TOPQ1.e6671_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.363355.Sherpa.DAOD_TOPQ1.e5525_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.363355.Sherpa.DAOD_TOPQ1.e5525_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.363355.Sherpa.DAOD_TOPQ1.e5525_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.363356.Sherpa.DAOD_TOPQ1.e5525_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.363356.Sherpa.DAOD_TOPQ1.e5525_s3126_r10201_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.363356.Sherpa.DAOD_TOPQ1.e5525_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.363357.Sherpa.DAOD_TOPQ1.e5525_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.363357.Sherpa.DAOD_TOPQ1.e5525_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.363357.Sherpa.DAOD_TOPQ1.e5525_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.363358.Sherpa.DAOD_TOPQ1.e5525_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.363358.Sherpa.DAOD_TOPQ1.e5525_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.363358.Sherpa.DAOD_TOPQ1.e5525_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.363359.Sherpa.DAOD_TOPQ1.e5583_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.363359.Sherpa.DAOD_TOPQ1.e5583_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.363359.Sherpa.DAOD_TOPQ1.e5583_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.363360.Sherpa.DAOD_TOPQ1.e5983_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.363360.Sherpa.DAOD_TOPQ1.e5983_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.363360.Sherpa.DAOD_TOPQ1.e5983_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.363489.Sherpa.DAOD_TOPQ1.e5525_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.363489.Sherpa.DAOD_TOPQ1.e5525_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.363489.Sherpa.DAOD_TOPQ1.e5525_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364100.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364100.Sherpa.DAOD_TOPQ1.e5271_s3126_r10201_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364100.Sherpa.DAOD_TOPQ1.e5271_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364101.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364101.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364101.Sherpa.DAOD_TOPQ1.e5271_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364102.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364102.Sherpa.DAOD_TOPQ1.e5271_s3126_r10201_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364102.Sherpa.DAOD_TOPQ1.e5271_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364103.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364103.Sherpa.DAOD_TOPQ1.e5271_s3126_r10201_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364103.Sherpa.DAOD_TOPQ1.e5271_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364104.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364104.Sherpa.DAOD_TOPQ1.e5271_s3126_r10201_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364104.Sherpa.DAOD_TOPQ1.e5271_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364105.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364105.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364105.Sherpa.DAOD_TOPQ1.e5271_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364106.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364106.Sherpa.DAOD_TOPQ1.e5271_s3126_r10201_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364106.Sherpa.DAOD_TOPQ1.e5271_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364107.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364107.Sherpa.DAOD_TOPQ1.e5271_s3126_r10201_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364107.Sherpa.DAOD_TOPQ1.e5271_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364108.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364108.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364108.Sherpa.DAOD_TOPQ1.e5271_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364109.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364109.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364109.Sherpa.DAOD_TOPQ1.e5271_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364110.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364110.Sherpa.DAOD_TOPQ1.e5271_s3126_r10201_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364110.Sherpa.DAOD_TOPQ1.e5271_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364111.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364111.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364111.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r9364_r9315_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364112.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364112.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364112.Sherpa.DAOD_TOPQ1.e5271_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364113.Sherpa.DAOD_TOPQ1.e5271_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364113.Sherpa.DAOD_TOPQ1.e5271_s3126_r10201_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364113.Sherpa.DAOD_TOPQ1.e5271_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364114.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364114.Sherpa.DAOD_TOPQ1.e5299_s3126_r10201_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364114.Sherpa.DAOD_TOPQ1.e5299_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364115.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364115.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364115.Sherpa.DAOD_TOPQ1.e5299_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364116.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364116.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364116.Sherpa.DAOD_TOPQ1.e5299_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364117.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364117.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364117.Sherpa.DAOD_TOPQ1.e5299_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364118.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364118.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364118.Sherpa.DAOD_TOPQ1.e5299_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364119.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364119.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364119.Sherpa.DAOD_TOPQ1.e5299_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364120.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364120.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364120.Sherpa.DAOD_TOPQ1.e5299_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364121.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364121.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364121.Sherpa.DAOD_TOPQ1.e5299_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364122.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364122.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364122.Sherpa.DAOD_TOPQ1.e5299_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364123.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364123.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364123.Sherpa.DAOD_TOPQ1.e5299_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364124.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364124.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364124.Sherpa.DAOD_TOPQ1.e5299_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364125.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364125.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364125.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r9364_r9315_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364126.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364126.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364126.Sherpa.DAOD_TOPQ1.e5299_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364127.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364127.Sherpa.DAOD_TOPQ1.e5299_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364127.Sherpa.DAOD_TOPQ1.e5299_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364128.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364128.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364128.Sherpa.DAOD_TOPQ1.e5307_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364129.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364129.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364129.Sherpa.DAOD_TOPQ1.e5307_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364130.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364130.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364130.Sherpa.DAOD_TOPQ1.e5307_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364131.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364131.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364131.Sherpa.DAOD_TOPQ1.e5307_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364132.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364132.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364132.Sherpa.DAOD_TOPQ1.e5307_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364133.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364133.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364133.Sherpa.DAOD_TOPQ1.e5307_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364134.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364134.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364134.Sherpa.DAOD_TOPQ1.e5307_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364135.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364135.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364135.Sherpa.DAOD_TOPQ1.e5307_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364136.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364136.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364136.Sherpa.DAOD_TOPQ1.e5307_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364137.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364137.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364137.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r9364_r9315_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364138.Sherpa.DAOD_TOPQ1.e5313_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364138.Sherpa.DAOD_TOPQ1.e5313_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364138.Sherpa.DAOD_TOPQ1.e5313_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364139.Sherpa.DAOD_TOPQ1.e5313_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364139.Sherpa.DAOD_TOPQ1.e5313_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364139.Sherpa.DAOD_TOPQ1.e5313_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364140.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364140.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364140.Sherpa.DAOD_TOPQ1.e5307_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364141.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364141.Sherpa.DAOD_TOPQ1.e5307_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364141.Sherpa.DAOD_TOPQ1.e5307_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364156.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364156.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364156.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364157.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364157.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364157.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r9364_r9315_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364158.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364158.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364158.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r9364_r9315_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364159.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364159.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364159.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364160.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364160.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364160.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364161.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364161.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364161.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364162.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364162.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364162.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364163.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364163.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364163.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r9364_r9315_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364164.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364164.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364164.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364165.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364165.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364165.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364166.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364166.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364166.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364167.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364167.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364167.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r9364_r9315_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364168.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364168.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364168.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364169.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364169.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364169.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364170.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364170.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364170.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364171.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364171.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364171.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r9364_r9315_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364172.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364172.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364172.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r9364_r9315_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364173.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364173.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364173.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364174.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364174.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364174.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364175.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364175.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364175.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r9364_r9315_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364176.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364176.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364176.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364177.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364177.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364177.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r9364_r9315_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364178.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364178.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364178.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364180.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364180.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364180.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364181.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364181.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364181.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r9364_r9315_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364182.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364182.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364182.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364183.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364183.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364183.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364184.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364184.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364184.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364185.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364185.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364185.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r9364_r9315_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364186.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364186.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364186.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364187.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364187.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364187.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364188.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364188.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364188.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364189.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364189.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364189.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364190.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364190.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r9364_r9315_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364190.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364191.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364191.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364191.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r9364_r9315_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364192.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364192.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364192.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364193.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364193.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364193.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r9364_r9315_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364194.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364194.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364194.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r9364_r9315_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364195.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364195.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364195.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364196.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364196.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_s3136_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364196.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364197.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364197.Sherpa.DAOD_TOPQ1.e5340_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364197.Sherpa.DAOD_TOPQ1.e5340_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364250.Sherpa.DAOD_TOPQ1.e5894_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364250.Sherpa.DAOD_TOPQ1.e5894_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364250.Sherpa.DAOD_TOPQ1.e5894_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364253.Sherpa.DAOD_TOPQ1.e5916_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364253.Sherpa.DAOD_TOPQ1.e5916_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364253.Sherpa.DAOD_TOPQ1.e5916_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364254.Sherpa.DAOD_TOPQ1.e5916_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364254.Sherpa.DAOD_TOPQ1.e5916_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364254.Sherpa.DAOD_TOPQ1.e5916_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.364255.Sherpa.DAOD_TOPQ1.e5916_e5984_s3126_r10201_r10210_p4512.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.364255.Sherpa.DAOD_TOPQ1.e5916_e5984_s3126_r10724_r10726_p4512.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.364255.Sherpa.DAOD_TOPQ1.e5916_s3126_r9364_p4512.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410464.aMcAtNloPy8EvtGen.DAOD_TOPQ1.e6762_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410464.aMcAtNloPy8EvtGen.DAOD_TOPQ1.e6762_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410464.aMcAtNloPy8EvtGen.DAOD_TOPQ1.e6762_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410465.aMcAtNloPy8EvtGen.DAOD_TOPQ1.e6762_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410465.aMcAtNloPy8EvtGen.DAOD_TOPQ1.e6762_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410465.aMcAtNloPy8EvtGen.DAOD_TOPQ1.e6762_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410560.MadGraphPythia8EvtGen.DAOD_TOPQ1.e5803_e5984_s3126_r10201_r10210_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410560.MadGraphPythia8EvtGen.DAOD_TOPQ1.e5803_e5984_s3126_r10724_r10726_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410560.MadGraphPythia8EvtGen.DAOD_TOPQ1.e5803_s3126_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410644.PowhegPythia8EvtGen.DAOD_TOPQ1.e6527_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410644.PowhegPythia8EvtGen.DAOD_TOPQ1.e6527_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410644.PowhegPythia8EvtGen.DAOD_TOPQ1.e6527_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410645.PowhegPythia8EvtGen.DAOD_TOPQ1.e6527_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410645.PowhegPythia8EvtGen.DAOD_TOPQ1.e6527_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410645.PowhegPythia8EvtGen.DAOD_TOPQ1.e6527_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410646.PowhegPythia8EvtGen.DAOD_TOPQ1.e6552_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410646.PowhegPythia8EvtGen.DAOD_TOPQ1.e6552_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410646.PowhegPythia8EvtGen.DAOD_TOPQ1.e6552_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410647.PowhegPythia8EvtGen.DAOD_TOPQ1.e6552_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410647.PowhegPythia8EvtGen.DAOD_TOPQ1.e6552_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410647.PowhegPythia8EvtGen.DAOD_TOPQ1.e6552_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410654.PowhegPythia8EvtGen.DAOD_TOPQ1.e6552_e5984_s3126_r10201_r10210_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410654.PowhegPythia8EvtGen.DAOD_TOPQ1.e6552_e5984_s3126_r9364_r9315_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410654.PowhegPythia8EvtGen.DAOD_TOPQ1.e6552_e5984_s3126_s3136_r10724_r10726_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410655.PowhegPythia8EvtGen.DAOD_TOPQ1.e6552_e5984_s3126_r10201_r10210_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410655.PowhegPythia8EvtGen.DAOD_TOPQ1.e6552_e5984_s3126_r9364_r9315_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410655.PowhegPythia8EvtGen.DAOD_TOPQ1.e6552_e5984_s3126_s3136_r10724_r10726_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410557.PowhegHerwig7EvtGen.DAOD_TOPQ1.e6366_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410557.PowhegHerwig7EvtGen.DAOD_TOPQ1.e6366_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410557.PowhegHerwig7EvtGen.DAOD_TOPQ1.e6366_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410558.PowhegHerwig7EvtGen.DAOD_TOPQ1.e6366_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410558.PowhegHerwig7EvtGen.DAOD_TOPQ1.e6366_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410558.PowhegHerwig7EvtGen.DAOD_TOPQ1.e6366_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.411032.PowhegHerwig7EvtGen.DAOD_TOPQ1.e6719_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.411032.PowhegHerwig7EvtGen.DAOD_TOPQ1.e6719_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.411032.PowhegHerwig7EvtGen.DAOD_TOPQ1.e6719_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.411033.PowhegHerwig7EvtGen.DAOD_TOPQ1.e6719_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.411033.PowhegHerwig7EvtGen.DAOD_TOPQ1.e6719_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.411033.PowhegHerwig7EvtGen.DAOD_TOPQ1.e6719_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.411036.PowhegHerwig7EvtGen.DAOD_TOPQ1.e6702_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.411036.PowhegHerwig7EvtGen.DAOD_TOPQ1.e6702_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.411036.PowhegHerwig7EvtGen.DAOD_TOPQ1.e6702_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.411037.PowhegHerwig7EvtGen.DAOD_TOPQ1.e6702_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.411037.PowhegHerwig7EvtGen.DAOD_TOPQ1.e6702_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.411037.PowhegHerwig7EvtGen.DAOD_TOPQ1.e6702_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.411034.PhHerwig7EG.DAOD_TOPQ1.e6734_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.411034.PhHerwig7EG.DAOD_TOPQ1.e6734_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.411034.PhHerwig7EG.DAOD_TOPQ1.e6734_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.411035.PhHerwig7EG.DAOD_TOPQ1.e6734_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.411035.PhHerwig7EG.DAOD_TOPQ1.e6734_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.411035.PhHerwig7EG.DAOD_TOPQ1.e6734_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.411082.PhHerwig7EG.DAOD_TOPQ1.e6799_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.411082.PhHerwig7EG.DAOD_TOPQ1.e6799_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.411082.PhHerwig7EG.DAOD_TOPQ1.e6799_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.411085.PhHerwig7EG.DAOD_TOPQ1.e6799_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.411085.PhHerwig7EG.DAOD_TOPQ1.e6799_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.411085.PhHerwig7EG.DAOD_TOPQ1.e6799_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.411086.PhHerwig7EG.DAOD_TOPQ1.e6799_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.411087.PhHerwig7EG.DAOD_TOPQ1.e6799_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.411087.PhHerwig7EG.DAOD_TOPQ1.e6799_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.411087.PhHerwig7EG.DAOD_TOPQ1.e6799_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.407348.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e6884_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.407348.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e6884_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.407348.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e6884_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.407349.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e6884_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.407349.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e6884_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.407349.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e6884_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.407350.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e6884_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.407350.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e6884_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.407350.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e6884_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410155.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e5070_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410155.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e5070_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410156.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e5070_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410156.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e5070_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410156.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e5070_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410157.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e5070_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410157.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e5070_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410157.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e5070_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410218.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e5070_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410218.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e5070_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410218.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e5070_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410219.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e5070_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410219.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e5070_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410219.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e5070_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.410220.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e5070_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.410220.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e5070_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.410220.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e5070_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.412002.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e6817_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.412002.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e6817_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.412002.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e6817_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.412005.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e6867_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.412005.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e6867_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.412005.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e6867_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.412043.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e7101_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.412043.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e7101_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.412043.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e7101_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.412004.aMcAtNloPy8EG.DAOD_TOPQ1.e6888_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.412004.aMcAtNloPy8EG.DAOD_TOPQ1.e6888_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.412004.aMcAtNloPy8EG.DAOD_TOPQ1.e6888_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.412066.aMcAtNloPy8EG.DAOD_TOPQ1.e7129_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.412066.aMcAtNloPy8EG.DAOD_TOPQ1.e7129_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.412066.aMcAtNloPy8EG.DAOD_TOPQ1.e7129_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.412067.aMcAtNloPy8EG.DAOD_TOPQ1.e7129_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.412067.aMcAtNloPy8EG.DAOD_TOPQ1.e7129_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.412067.aMcAtNloPy8EG.DAOD_TOPQ1.e7129_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.412068.aMcAtNloPy8EG.DAOD_TOPQ1.e7129_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.412068.aMcAtNloPy8EG.DAOD_TOPQ1.e7129_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.412068.aMcAtNloPy8EG.DAOD_TOPQ1.e7129_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.412069.aMcAtNloPy8EG.DAOD_TOPQ1.e7129_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.412069.aMcAtNloPy8EG.DAOD_TOPQ1.e7129_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.412069.aMcAtNloPy8EG.DAOD_TOPQ1.e7129_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.412070.aMcAtNloPy8EG.DAOD_TOPQ1.e7129_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.412070.aMcAtNloPy8EG.DAOD_TOPQ1.e7129_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.412070.aMcAtNloPy8EG.DAOD_TOPQ1.e7129_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"user.tnommens.412071.aMcAtNloPy8EG.DAOD_TOPQ1.e7129_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root",
    root+"user.tnommens.412071.aMcAtNloPy8EG.DAOD_TOPQ1.e7129_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root",
    root+"user.tnommens.412071.aMcAtNloPy8EG.DAOD_TOPQ1.e7129_a875_r9364_p4514.bsm4t_gnn_mc16a_output_root",
    root+"ttH_tttt_m1000",
#    root+"ttH_tttt_m900",
#    root+"ttH_tttt_m800",
#    root+"ttH_tttt_m700",
#    root+"ttH_tttt_m600",
#    root+"ttH_tttt_m500",
#    root+"ttH_tttt_m400",
]


def mapping(name):
    if "_singletop_"  in name: return "$t$"
    if "_tchan_"      in name: return "$t$"
    if "_Wt_"         in name: return "$Wt$"
    if "_ttbarHT1k_"  in name: return "$t\\bar{t}$"
    if "_ttbar_"      in name: return "$t\\bar{t}$"
    if "_ttbarHT1k5_" in name: return "$t\\bar{t}$"
    if "_ttbarHT6c_"  in name: return "$t\\bar{t}$"
    if "_ttH125_"     in name: return "$t\\bar{t}H$"
    if "_SM4topsNLO"  in name: return "$t\\bar{t}t\\bar{t}$"
    if "_tt_"         in name: return "$t\\bar{t}$"
    if "_ttee."       in name: return "$t\\bar{t}\\ell\\ell$"
    if "_ttmumu."     in name: return "$t\\bar{t}\\ell\\ell$"
    if "_tttautau."   in name: return "$t\\bar{t}\\ell\\ell$"
    if "_ttW."        in name: return "$t\\bar{t}V$"
    if "_ttZnunu."    in name: return "$t\\bar{t}V$"
    if "_ttZqq."      in name: return "$t\\bar{t}V$"
    if "_tW."         in name: return "$tV$"
    if "_tW_"         in name: return "$tV$"
    if "_tZ."         in name: return "$tV$"
    if "_WH125."      in name: return "$VH$"
    if "_WH125_"      in name: return "$VH$"
    if "_ZH125_"      in name: return "$VH$"
    if "_WplvWmqq"    in name: return "$V_{1}V_{2}$"
    if "_WpqqWmlv"    in name: return "$V_{1}V_{2}$"
    if "_WlvZqq"      in name: return "$V_{1}V_{2}$"
    if "_WqqZll"      in name: return "$V_{1}V_{2}$"
    if "_WqqZvv"      in name: return "$V_{1}V_{2}$"
    if "_ZqqZll"      in name: return "$V_{1}V_{2}$"
    if "_ZqqZvv"      in name: return "$V_{1}V_{2}$"
    if "_llll"        in name: return "$\\ell,\\nu$"
    if "_lllv"        in name: return "$\\ell,\\nu$"
    if "_llvv"        in name: return "$\\ell,\\nu$"
    if "_lvvv"        in name: return "$\\ell,\\nu$"
    if "_Wmunu_"      in name: return "$V\\ell\\nu$"
    if "_Wenu_"       in name: return "$V\\ell\\nu$"
    if "_Wtaunu_"     in name: return "$V\\ell\\nu$"
    if "_Zee_"        in name: return "$V\\ell\\ell$"
    if "_Ztautau_"    in name: return "$V\\ell\\ell$"
    if "_Zmumu_"      in name: return "$V\\ell\\ell$"
    if "ttH_tttt"     in name: return "$t\\bar{t}t\\bar{t}H_{" + name.split("tttt_m")[-1].split(".")[0] + "}$"
    print("----> " + name)
    exit()

class container:
    def __init__(self):
        self.proc = ""
        self._sow_nominal = []
        self._plot_data   = []
        self._weights     = None
        self._meta        = None
        self.lint_atlas   = 140.1

    def __add__(self, other):
        self._weights     += other.sow_weights
        self._sow_nominal += other._sow_nominal
        self._plot_data   += other._plot_data
        return self

    def __radd__(self, other):
        if not isinstance(other, int): return self.__add__(other)
        c = container()
        c._meta = self._meta
        c.proc  = self.proc
        c._weights = 0
        return c.__add__(self)

    @property
    def expected_events(self): return (self.cross_section*self.lint_atlas)

    @property
    def scale_factor(self): return self.expected_events / self.sow_weights

    @property
    def cross_section(self): return self._meta.crossSection

    @property
    def sow_weights(self):
        if self._weights is not None: return self._weights
        return self._meta.SumOfWeights[b"sumWeights"]["total_events_weighted"]

    @property
    def num_events(self): return len(self._sow_nominal)

    @property
    def DatasetName(self): return self._meta.DatasetName

    def rescale(self):
        s = self.scale_factor
        self._sow_nominal = [i*s for i in self._sow_nominal]

    @property
    def hist(self):
        th = TH1F()
        th.Weights = self._sow_nominal
        th.xData   = self._plot_data
        th.Title   = self.proc
        return th

def compute_data(io_handle, meta):
    stacks = {}
    for i in io_handle:
        fname = i["filename"].decode("utf-8")
        try: c = stacks[fname]
        except:
            m = None
            c = container()
            for h, k in zip(meta.keys(), meta.values()):
                if k.hash(fname) not in h: continue
                m = k
                break
            if m is None: print("-> ", fname); exit()
            c.proc          = m.DatasetName
            c._meta         = m
            stacks[fname]   = c
        c._plot_data.append(i[b"nominal.met_met.met_met"]/1000)
        c._sow_nominal.append(i[b"nominal.weight_mc.weight_mc"])
    return stacks

def buffs(tl, val, tm):
    if isinstance(val, float): val = round(val, 5)
    l1, v1 = len(tl), str(val)
    return tm + "".join([" "]*(l1 - len(v1))) + v1 + " | "



x = Analysis()
x.FetchMeta = True
#x.SumOfWeightsTreeName = "sumWeights"
#x.AddSamples(root, "data")
x.Start()
mtx = x.GetMetaData

gxt = []
for i in sets:
    msk = False
    msk += "ttH_tttt" in i
    msk += "Pythia8EvtGen" in i
    msk += "PhPy8EG" in i
    msk += "Sherpa" in i
    msk += "aMcAtNloPy8EvtGen" in i
    msk += "MadGraphPythia8EvtGen" in i
    msk += "PowhegPythia8EvtGen" in i
    msk += "PowhegHerwig7EvtGen" in i
    msk += "PhHerwig7EG" in i
    msk += "aMcAtNloPythia8EvtGen" in i
    msk += "aMcAtNloPy8EG" in i
    if not msk: continue
    gxt.append(i+"/*")

smpl = IO(gxt)
smpl.Verbose = False
smpl.Trees = ["nominal"]
smpl.Leaves = ["weight_mc", "met_met"]
opt = compute_data(smpl, mtx)

proc = {}
for i in opt:
    c = opt[i]
    prc = c.DatasetName
    if prc not in proc: proc[prc] = []
    proc[prc] += [c]

tmp = {}
for i in proc:
    proc[i] = sum(proc[i])
    proc[i].rescale()
    pc = mapping(proc[i].DatasetName)
    if pc not in tmp: tmp[pc] = []
    tmp[pc] += [proc[i]]

proc.clear()
for i in tmp:
    c = sum(tmp[i])
    c.proc = i
    entries = float(sum(c.hist.counts))
    proc[entries] = c

titles = [
    "Sample Processed           ",
    "Exp. Yield (lumi*x-section)",
    "x-section (fb) ",
    "Num. Events (unweighted)",
    "Sum of Weights (Tree)",
    "Scale factor (Exp. yield / sum of weights)",
    "Yield (Scale factor * histogram)",
    ""
]

h = []
print(" | ".join(titles))
for i in list(sorted(proc)):
    i = proc[i]
    prx = i.proc
    hx  = i.hist

    tmp = ""
    tmp = buffs(titles[0], prx, tmp)
    tmp = buffs(titles[1], i.expected_events, tmp)
    tmp = buffs(titles[2], i.cross_section, tmp)
    tmp = buffs(titles[3], i.num_events, tmp)
    tmp = buffs(titles[4], i.sow_weights, tmp)
    tmp = buffs(titles[5], i.scale_factor, tmp)
    tmp = buffs(titles[6], float(sum(hx.counts)), tmp)
    h.append(hx)
    print(tmp)

comb = TH1F()
comb.Stacked = True
comb.yScaling = 5*4.8
comb.xScaling = 5*6.4
comb.Histograms = h
comb.LegendSize = 0.1
comb.Title  = "Weighted Missing Transverse Energy"
comb.xTitle = "Missing ET (GeV)"
comb.yTitle = "Events / (4 GeV)"
comb.xBins = 250
comb.xMin = 0
comb.xMax = 1000
comb.yMin = 0.1
comb.xStep = 100
comb.yLogarithmic = True
comb.Filename = "weighted_hists"
comb.SaveFigure()


