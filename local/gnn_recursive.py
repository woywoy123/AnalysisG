from AnalysisG.core.lossfx import OptimizerConfig
from AnalysisG.generators.analysis import Analysis
from AnalysisG.events.event_bsm_4tops import BSM4Tops
from AnalysisG.graphs.graph_bsm_4tops import GraphTops, GraphChildren
from AnalysisG.models.RecursiveGraphNeuralNetwork import *

smpls = [
    "mc16_13TeV.304014.MadGraphPythia8EvtGen_A14NNPDF23_3top_SM.deriv.DAOD_TOPQ1.e4324_s3126_r10201_p4514",
    "mc16_13TeV.304014.MadGraphPythia8EvtGen_A14NNPDF23_3top_SM.deriv.DAOD_TOPQ1.e4324_s3126_r10724_p4514",
    "mc16_13TeV.304014.MadGraphPythia8EvtGen_A14NNPDF23_3top_SM.deriv.DAOD_TOPQ1.e4324_s3126_r9364_p4514",
    "mc16_13TeV.312440.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m400.deriv.DAOD_TOPQ1.e7743_a875_r10201_p4031",
    "mc16_13TeV.312440.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m400.deriv.DAOD_TOPQ1.e7743_a875_r10724_p4031",
    "mc16_13TeV.312440.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m400.deriv.DAOD_TOPQ1.e7743_a875_r9364_p4031",
    "mc16_13TeV.312441.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m500.deriv.DAOD_TOPQ1.e7743_a875_r10201_p4031",
    "mc16_13TeV.312441.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m500.deriv.DAOD_TOPQ1.e7743_a875_r10724_p4031",
    "mc16_13TeV.312441.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m500.deriv.DAOD_TOPQ1.e7743_a875_r9364_p4031",
    "mc16_13TeV.312442.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m600.deriv.DAOD_TOPQ1.e7743_a875_r10201_p4031",
    "mc16_13TeV.312442.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m600.deriv.DAOD_TOPQ1.e7743_a875_r9364_p4031",
    "mc16_13TeV.312443.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m700.deriv.DAOD_TOPQ1.e7743_a875_r10201_p4031",
    "mc16_13TeV.312443.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m700.deriv.DAOD_TOPQ1.e7743_a875_r10724_p4031",
    "mc16_13TeV.312443.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m700.deriv.DAOD_TOPQ1.e7743_a875_r9364_p4031",
    "mc16_13TeV.312444.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m800.deriv.DAOD_TOPQ1.e7743_a875_r10201_p4031",
    "mc16_13TeV.312444.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m800.deriv.DAOD_TOPQ1.e7743_a875_r10724_p4031",
    "mc16_13TeV.312444.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m800.deriv.DAOD_TOPQ1.e7743_a875_r9364_p4031",
    "mc16_13TeV.312445.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m900.deriv.DAOD_TOPQ1.e7743_a875_r10201_p4031",
    "mc16_13TeV.312445.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m900.deriv.DAOD_TOPQ1.e7743_a875_r10724_p4031",
    "mc16_13TeV.312445.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m900.deriv.DAOD_TOPQ1.e7743_a875_r9364_p4031",
    "mc16_13TeV.312446.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m1000.deriv.DAOD_TOPQ1.e7743_a875_r10201_p4031",
    "mc16_13TeV.312446.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m1000.deriv.DAOD_TOPQ1.e7743_a875_r10724_p4031",
    "mc16_13TeV.312446.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m1000.deriv.DAOD_TOPQ1.e7743_a875_r9364_p4031",
    "mc16_13TeV.407342.PhPy8EG_A14_ttbarHT1k5_hdamp258p75_nonallhad.deriv.DAOD_TOPQ1.e6414_a875_r9364_p4514",
    "mc16_13TeV.407342.PhPy8EG_A14_ttbarHT1k5_hdamp258p75_nonallhad.deriv.DAOD_TOPQ1.e6414_s3126_r9364_p4512",
    "mc16_13TeV.407343.PhPy8EG_A14_ttbarHT1k_1k5_hdamp258p75_nonallhad.deriv.DAOD_TOPQ1.e6414_a875_r9364_p4514",
    "mc16_13TeV.407343.PhPy8EG_A14_ttbarHT1k_1k5_hdamp258p75_nonallhad.deriv.DAOD_TOPQ1.e6414_s3126_r9364_p4512",
    "mc16_13TeV.407344.PhPy8EG_A14_ttbarHT6c_1k_hdamp258p75_nonallhad.deriv.DAOD_TOPQ1.e6414_a875_r9364_p4514",
    "mc16_13TeV.407348.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttbarHT1k5_nonAH.deriv.DAOD_TOPQ1.e6884_a875_r9364_p4514",
    "mc16_13TeV.407349.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttbarHT1k_1k5_nonAH.deriv.DAOD_TOPQ1.e6884_a875_r9364_p4514",
    "mc16_13TeV.407350.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttbarHT6c_1k_nonAH.deriv.DAOD_TOPQ1.e6884_a875_r9364_p4514",
    "mc16_13TeV.407354.PhH7EG_H7UE_ttbarHT1k5_hdamp258p75_704_nonAH.deriv.DAOD_TOPQ1.e6894_a875_r9364_p4514",
    "mc16_13TeV.407355.PhH7EG_H7UE_ttbarHT1k_1k5_hdamp258p75_704_nonAH.deriv.DAOD_TOPQ1.e6894_a875_r9364_p4514",
    "mc16_13TeV.407356.PhH7EG_H7UE_ttbarHT6c_1k_hdamp258p75_704_nonAH.deriv.DAOD_TOPQ1.e6902_a875_r9364_p4514",
    "mc16_13TeV.410081.MadGraphPythia8EvtGen_A14NNPDF23_ttbarWW.deriv.DAOD_TOPQ1.e4111_s3126_r10201_p4514",
    "mc16_13TeV.410081.MadGraphPythia8EvtGen_A14NNPDF23_ttbarWW.deriv.DAOD_TOPQ1.e4111_s3126_r10724_p4514",
    "mc16_13TeV.410081.MadGraphPythia8EvtGen_A14NNPDF23_ttbarWW.deriv.DAOD_TOPQ1.e4111_s3126_r9364_p4514",
    "mc16_13TeV.410155.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttW.deriv.DAOD_TOPQ1.e5070_s3126_r10201_p4514",
    "mc16_13TeV.410155.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttW.deriv.DAOD_TOPQ1.e5070_s3126_r10724_p4514",
    "mc16_13TeV.410155.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttW.deriv.DAOD_TOPQ1.e5070_s3126_r9364_p4514",
    "mc16_13TeV.410156.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttZnunu.deriv.DAOD_TOPQ1.e5070_s3126_r10201_p4514",
    "mc16_13TeV.410156.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttZnunu.deriv.DAOD_TOPQ1.e5070_s3126_r10724_p4514",
    "mc16_13TeV.410156.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttZnunu.deriv.DAOD_TOPQ1.e5070_s3126_r9364_p4514",
    "mc16_13TeV.410157.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttZqq.deriv.DAOD_TOPQ1.e5070_s3126_r10201_p4514",
    "mc16_13TeV.410157.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttZqq.deriv.DAOD_TOPQ1.e5070_s3126_r10724_p4514",
    "mc16_13TeV.410157.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttZqq.deriv.DAOD_TOPQ1.e5070_s3126_r9364_p4514",
    "mc16_13TeV.410218.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttee.deriv.DAOD_TOPQ1.e5070_s3126_r10201_p4514",
    "mc16_13TeV.410218.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttee.deriv.DAOD_TOPQ1.e5070_s3126_r10724_p4514",
    "mc16_13TeV.410218.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttee.deriv.DAOD_TOPQ1.e5070_s3126_r9364_p4514",
    "mc16_13TeV.410219.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttmumu.deriv.DAOD_TOPQ1.e5070_s3126_r9364_p4514",
    "mc16_13TeV.410220.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_tttautau.deriv.DAOD_TOPQ1.e5070_s3126_r9364_p4514",
    "mc16_13TeV.410408.aMcAtNloPythia8EvtGen_tWZ_Ztoll_minDR1.deriv.DAOD_TOPQ1.e6423_s3126_r9364_p4514",
    "mc16_13TeV.410464.aMcAtNloPy8EvtGen_MEN30NLO_A14N23LO_ttbar_noShWe_SingleLep.deriv.DAOD_TOPQ1.e6762_a875_r9364_p4514",
    "mc16_13TeV.410465.aMcAtNloPy8EvtGen_MEN30NLO_A14N23LO_ttbar_noShWe_dil.deriv.DAOD_TOPQ1.e6762_a875_r9364_p4514",
    "mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_TOPQ1.e6337_s3126_r10201_p4514",
    "mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_TOPQ1.e6337_s3126_r10724_p4514",
    "mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_TOPQ1.e6337_s3126_r9364_p4514",
    "mc16_13TeV.410480.PhPy8EG_A14_ttbar_hdamp517p5_SingleLep.deriv.DAOD_TOPQ1.e6454_a875_r9364_p4514",
    "mc16_13TeV.410482.PhPy8EG_A14_ttbar_hdamp517p5_dil.deriv.DAOD_TOPQ1.e6454_a875_r9364_p4514",
    "mc16_13TeV.410557.PowhegHerwig7EvtGen_H7UE_tt_hdamp258p75_704_SingleLep.deriv.DAOD_TOPQ1.e6366_a875_r9364_p4514",
    "mc16_13TeV.410558.PowhegHerwig7EvtGen_H7UE_tt_hdamp258p75_704_dil.deriv.DAOD_TOPQ1.e6366_a875_r9364_p4514",
    "mc16_13TeV.410560.MadGraphPythia8EvtGen_A14_tZ_4fl_tchan_noAllHad.deriv.DAOD_TOPQ1.e5803_s3126_r9364_p4514",
    "mc16_13TeV.410644.PowhegPythia8EvtGen_A14_singletop_schan_lept_top.deriv.DAOD_TOPQ1.e6527_a875_r9364_p4514",
    "mc16_13TeV.410644.PowhegPythia8EvtGen_A14_singletop_schan_lept_top.deriv.DAOD_TOPQ1.e6527_s3126_r9364_p4514",
    "mc16_13TeV.410645.PowhegPythia8EvtGen_A14_singletop_schan_lept_antitop.deriv.DAOD_TOPQ1.e6527_a875_r9364_p4514",
    "mc16_13TeV.410645.PowhegPythia8EvtGen_A14_singletop_schan_lept_antitop.deriv.DAOD_TOPQ1.e6527_s3126_r9364_p4514"
]

x = BSM4Tops()
tt = GraphChildren()
m = RecursiveGraphNeuralNetwork()
m.o_edge = {"top_edge" : "CrossEntropyLoss"}
m.i_node = ["pt", "eta", "phi", "energy"]
m.i_graph = ["met", "phi"]
m.device = "cuda:1"
op = OptimizerConfig()
op.Optimizer = "adam"
op.lr = 1e-2

root1 = "/scratch/tnom6927/samples/"

ana = Analysis()
for i in smpls:
    ana.AddSamples(root1 + i + "/*", i)
    ana.AddEvent(x, i)
    ana.AddGraph(tt, i)
ana.AddModel(m, op, "rnn-mrk1")
ana.kFolds = 10
ana.Epochs = 100
ana.Start()
