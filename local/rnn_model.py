from AnalysisG.core.lossfx import OptimizerConfig
from AnalysisG.generators.analysis import Analysis
from AnalysisG.events.event_bsm_4tops import BSM4Tops
from AnalysisG.graphs.graph_bsm_4tops import GraphTops, GraphChildren, GraphTruthJets
from AnalysisG.models.RecursiveGraphNeuralNetwork import *

smpls = [
    "mc16_13TeV.304014.MadGraphPythia8EvtGen_A14NNPDF23_3top_SM.deriv.DAOD_TOPQ1.e4324_s3126_r10201_p4514",
#    "mc16_13TeV.304014.MadGraphPythia8EvtGen_A14NNPDF23_3top_SM.deriv.DAOD_TOPQ1.e4324_s3126_r10724_p4514",
#    "mc16_13TeV.304014.MadGraphPythia8EvtGen_A14NNPDF23_3top_SM.deriv.DAOD_TOPQ1.e4324_s3126_r9364_p4514",
#    "mc16_13TeV.312440.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m400.deriv.DAOD_TOPQ1.e7743_a875_r10201_p4031",
#    "mc16_13TeV.312440.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m400.deriv.DAOD_TOPQ1.e7743_a875_r10724_p4031",
#    "mc16_13TeV.312440.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m400.deriv.DAOD_TOPQ1.e7743_a875_r9364_p4031",
#    "mc16_13TeV.312441.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m500.deriv.DAOD_TOPQ1.e7743_a875_r10201_p4031",
#    "mc16_13TeV.312441.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m500.deriv.DAOD_TOPQ1.e7743_a875_r10724_p4031",
#    "mc16_13TeV.312441.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m500.deriv.DAOD_TOPQ1.e7743_a875_r9364_p4031",
#    "mc16_13TeV.312442.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m600.deriv.DAOD_TOPQ1.e7743_a875_r10201_p4031",
#    "mc16_13TeV.312442.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m600.deriv.DAOD_TOPQ1.e7743_a875_r9364_p4031",
#    "mc16_13TeV.312443.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m700.deriv.DAOD_TOPQ1.e7743_a875_r10201_p4031",
#    "mc16_13TeV.312443.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m700.deriv.DAOD_TOPQ1.e7743_a875_r10724_p4031",
#    "mc16_13TeV.312443.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m700.deriv.DAOD_TOPQ1.e7743_a875_r9364_p4031",
#    "mc16_13TeV.312444.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m800.deriv.DAOD_TOPQ1.e7743_a875_r10201_p4031",
#    "mc16_13TeV.312444.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m800.deriv.DAOD_TOPQ1.e7743_a875_r10724_p4031",
#    "mc16_13TeV.312444.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m800.deriv.DAOD_TOPQ1.e7743_a875_r9364_p4031",
#    "mc16_13TeV.312445.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m900.deriv.DAOD_TOPQ1.e7743_a875_r10201_p4031",
#    "mc16_13TeV.312445.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m900.deriv.DAOD_TOPQ1.e7743_a875_r10724_p4031",
#    "mc16_13TeV.312445.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m900.deriv.DAOD_TOPQ1.e7743_a875_r9364_p4031",
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
#    "mc16_13TeV.410155.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttW.deriv.DAOD_TOPQ1.e5070_s3126_r10201_p4514",
#    "mc16_13TeV.410155.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttW.deriv.DAOD_TOPQ1.e5070_s3126_r10724_p4514",
#    "mc16_13TeV.410155.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttW.deriv.DAOD_TOPQ1.e5070_s3126_r9364_p4514",
#    "mc16_13TeV.410156.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttZnunu.deriv.DAOD_TOPQ1.e5070_s3126_r10201_p4514",
#    "mc16_13TeV.410156.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttZnunu.deriv.DAOD_TOPQ1.e5070_s3126_r10724_p4514",
#    "mc16_13TeV.410156.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttZnunu.deriv.DAOD_TOPQ1.e5070_s3126_r9364_p4514",
#    "mc16_13TeV.410157.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttZqq.deriv.DAOD_TOPQ1.e5070_s3126_r10201_p4514",
#    "mc16_13TeV.410157.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttZqq.deriv.DAOD_TOPQ1.e5070_s3126_r10724_p4514",
#    "mc16_13TeV.410157.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttZqq.deriv.DAOD_TOPQ1.e5070_s3126_r9364_p4514",
#    "mc16_13TeV.410218.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttee.deriv.DAOD_TOPQ1.e5070_s3126_r10201_p4514",
#    "mc16_13TeV.410218.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttee.deriv.DAOD_TOPQ1.e5070_s3126_r10724_p4514",
#    "mc16_13TeV.410218.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttee.deriv.DAOD_TOPQ1.e5070_s3126_r9364_p4514",
#    "mc16_13TeV.410219.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttmumu.deriv.DAOD_TOPQ1.e5070_s3126_r9364_p4514",
#    "mc16_13TeV.410220.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_tttautau.deriv.DAOD_TOPQ1.e5070_s3126_r9364_p4514",
#    "mc16_13TeV.410408.aMcAtNloPythia8EvtGen_tWZ_Ztoll_minDR1.deriv.DAOD_TOPQ1.e6423_s3126_r9364_p4514",
    "mc16_13TeV.410464.aMcAtNloPy8EvtGen_MEN30NLO_A14N23LO_ttbar_noShWe_SingleLep.deriv.DAOD_TOPQ1.e6762_a875_r9364_p4514",
    "mc16_13TeV.410465.aMcAtNloPy8EvtGen_MEN30NLO_A14N23LO_ttbar_noShWe_dil.deriv.DAOD_TOPQ1.e6762_a875_r9364_p4514",
    "mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_TOPQ1.e6337_s3126_r10201_p4514",
    "mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_TOPQ1.e6337_s3126_r10724_p4514",
    "mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_TOPQ1.e6337_s3126_r9364_p4514",
    "mc16_13TeV.410480.PhPy8EG_A14_ttbar_hdamp517p5_SingleLep.deriv.DAOD_TOPQ1.e6454_a875_r9364_p4514",
    "mc16_13TeV.410482.PhPy8EG_A14_ttbar_hdamp517p5_dil.deriv.DAOD_TOPQ1.e6454_a875_r9364_p4514",
#    "mc16_13TeV.410557.PowhegHerwig7EvtGen_H7UE_tt_hdamp258p75_704_SingleLep.deriv.DAOD_TOPQ1.e6366_a875_r9364_p4514",
#    "mc16_13TeV.410558.PowhegHerwig7EvtGen_H7UE_tt_hdamp258p75_704_dil.deriv.DAOD_TOPQ1.e6366_a875_r9364_p4514",
#    "mc16_13TeV.410560.MadGraphPythia8EvtGen_A14_tZ_4fl_tchan_noAllHad.deriv.DAOD_TOPQ1.e5803_s3126_r9364_p4514",
#    "mc16_13TeV.410644.PowhegPythia8EvtGen_A14_singletop_schan_lept_top.deriv.DAOD_TOPQ1.e6527_a875_r9364_p4514",
#    "mc16_13TeV.410644.PowhegPythia8EvtGen_A14_singletop_schan_lept_top.deriv.DAOD_TOPQ1.e6527_s3126_r9364_p4514",
#    "mc16_13TeV.410645.PowhegPythia8EvtGen_A14_singletop_schan_lept_antitop.deriv.DAOD_TOPQ1.e6527_a875_r9364_p4514",
#    "mc16_13TeV.410645.PowhegPythia8EvtGen_A14_singletop_schan_lept_antitop.deriv.DAOD_TOPQ1.e6527_s3126_r9364_p4514"
]



params = [
#    ("MRK-1", "adam", {"lr" : 1e-6}),
#    ("MRK-2", "adam", {"lr" : 1e-8}),
    ("MRK-3", "adam", {"lr" : 1e-6, "amsgrad" : True}),
#    ("MRK-4", "sgd", {"lr" : 1e-6}),
#    ("MRK-5", "sgd", {"lr" : 1e-8}),
#    ("MRK-6", "sgd", {"lr" : 1e-5, "momentum" : 0.1}),
#    ("MRK-7", "sgd", {"lr" : 1e-5, "momentum" : 0.05, "dampening" : 0.01})
]

trains = []
optims = []
for k in params:
    m1 = RecursiveGraphNeuralNetwork()
    m1.o_edge  = {"top_edge" : "CrossEntropyLoss"}
    m1.i_node  = ["pt", "eta", "phi", "energy", "is_lep", "is_b"]
    m1.i_graph = ["met", "phi"]
    m1.device  = "cuda:0"

    opti = OptimizerConfig()
    opti.Optimizer = k[1]
    for t in k[2]: setattr(opti, t, k[2][t])

    trains.append(m1)
    optims.append(opti)


event = BSM4Tops()
graph = GraphTruthJets()
#graph = GraphChildren()

outs = "/eos/home-t/tnommens/mc16-update/Dilepton/"
ana = Analysis()
ana.OutputPath = "./RecursiveGNN-truthjets/"

for i in smpls:
    ana.AddSamples(outs + i + "/*", i)
    ana.AddEvent(event, i)
    ana.AddGraph(graph, i)

for i in range(len(optims)): ana.AddModel(trains[i], optims[i], params[i][0])

ana.kFolds = 1
#ana.kFold = [0, 1]
ana.Epochs = 250
ana.MaxRange = 500
#ana.TrainSize = 60
ana.Evaluation = False
ana.ContinueTraining = True
ana.TrainingDataset = "./RecursiveGNN-truthjets/dataset"
ana.Targets = ["top_edge"]
ana.Start()
