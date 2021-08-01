# Event Generator closure test
from Functions.Event.Event import EventGenerator
import uproot
from Functions.Particles.Particles import *
from Functions.Event.Event import Event

def ManualUproot(dir, Tree, Branch):
    F = uproot.open(dir)
    en = F[Tree + "/" + Branch].array(library = "np")
    return en    

def Comparison(dir, Tree, Branch, Events,Spawned):
    el = Electron()
    mu = Muon()
    truthjet = TruthJet()
    jet = Jet()
    top = Top()
    child = Truth_Top_Child()
    child_init = Truth_Top_Child_Init()
    event = Event()

    en = ManualUproot(dir, Tree, Branch)
    if Events == -1:
        Events = len(en)
    for i in range(Events):
        f = en[i]
        Ev = Spawned[i][Tree]

        
        comp = False 
        try: 
            len(f)
        except:
            if Branch in event.KeyMap:
                Compare = Ev.GetAttributeFromKeyMap(Branch)
                comp = True
                if Compare == f:
                    assert Compare == f
                else:
                    print("FAILURE::" + Branch + " " + Tree + " ")
                    print(Compare, f) 
                    assert Compare == f

            continue
 
        for x in range(len(f)):
            Compare = ""
            if Branch in el.KeyMap:
                Compare = Ev.Electrons[x].GetAttributeFromKeyMap(Branch)
            
            if Branch in mu.KeyMap:
                Compare = Ev.Muons[x].GetAttributeFromKeyMap(Branch)
            
            if Branch in truthjet.KeyMap:
                Compare = Ev.TruthJets[x].GetAttributeFromKeyMap(Branch)
            
            if Branch in jet.KeyMap:
                Compare = Ev.Jets[x].GetAttributeFromKeyMap(Branch)
            
            if Branch in top.KeyMap:
                Compare = Ev.TruthTops[x].GetAttributeFromKeyMap(Branch)
            


          
            try:
                float(f[x])
                listed = False
                if Compare == f[x] and listed == False:
                    assert Compare == f[x]
                    comp = True
                else:
                   print("FAILURE::" + Branch + " " + Tree + " " + str(x), " " + str(f[x]))
                   assert Compare == f[x] 
            except:
                listed = True

            if listed == True:
                for k in range(len(f[x])):
                    if Branch in child.KeyMap:
                        Compare = Ev.TruthChildren[x][k].GetAttributeFromKeyMap(Branch)
                    
                    if Branch in child_init.KeyMap:
                        Compare = Ev.TruthChildren_init[x][k].GetAttributeFromKeyMap(Branch)

                    if Compare != f[x][k]:
                        print("FAILURE::" + Branch + " " + Tree + " " + str(x) + " " + str(k) + " " + Compare)
                        assert Compare == f[x][k]
                    comp = True

            if comp:
                pass
            else:
                print("FAILURE::" + Branch + " " + Tree + " " + str(x) )

    print("PASSED:: " + Branch + " " + Tree)





dir = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root"


def TestParticleAssignment():
    Events = -1
    x = EventGenerator(dir, DebugThresh = Events)
    x.SpawnEvents()
    
    #Electrons 
    Comparison(dir, "nominal", "el_pt", Events, x.Events)
    Comparison(dir, "nominal", "el_e", Events, x.Events)
    Comparison(dir, "nominal", "el_phi", Events, x.Events)
    Comparison(dir, "nominal", "el_eta", Events, x.Events)
    Comparison(dir, "nominal", "el_charge", Events, x.Events)
    Comparison(dir, "nominal", "el_topoetcone20", Events, x.Events)
    Comparison(dir, "nominal", "el_ptvarcone20", Events, x.Events)
    Comparison(dir, "nominal", "el_CF", Events, x.Events)
    Comparison(dir, "nominal", "el_d0sig", Events, x.Events)
    Comparison(dir, "nominal", "el_delta_z0_sintheta", Events, x.Events)
    Comparison(dir, "nominal", "el_true_type", Events, x.Events)
    Comparison(dir, "nominal", "el_true_origin", Events, x.Events)
    Comparison(dir, "nominal", "el_true_firstEgMotherTruthType", Events, x.Events)
    Comparison(dir, "nominal", "el_true_firstEgMotherTruthOrigin", Events, x.Events)
    Comparison(dir, "nominal", "el_true_firstEgMotherPdgId", Events, x.Events)
    Comparison(dir, "nominal", "el_true_IFFclass", Events, x.Events)
    Comparison(dir, "nominal", "el_true_isPrompt", Events, x.Events)
    Comparison(dir, "nominal", "el_true_isChargeFl", Events, x.Events)
    
    #Muons
    Comparison(dir, "nominal", "mu_pt", Events, x.Events)
    Comparison(dir, "nominal", "mu_e", Events, x.Events)
    Comparison(dir, "nominal", "mu_phi", Events, x.Events)
    Comparison(dir, "nominal", "mu_eta", Events, x.Events)
    Comparison(dir, "nominal", "mu_charge", Events, x.Events)
    Comparison(dir, "nominal", "mu_topoetcone20", Events, x.Events)
    Comparison(dir, "nominal", "mu_ptvarcone30", Events, x.Events)
    Comparison(dir, "nominal", "mu_d0sig", Events, x.Events)
    Comparison(dir, "nominal", "mu_delta_z0_sintheta", Events, x.Events)
    Comparison(dir, "nominal", "mu_true_type", Events, x.Events)
    Comparison(dir, "nominal", "mu_true_origin", Events, x.Events)
    Comparison(dir, "nominal", "mu_true_IFFclass", Events, x.Events)
    Comparison(dir, "nominal", "mu_true_isPrompt", Events, x.Events)

    #jets
    Comparison(dir, "nominal", "jet_pt", Events, x.Events)
    Comparison(dir, "nominal", "jet_e", Events, x.Events)
    Comparison(dir, "nominal", "jet_phi", Events, x.Events)
    Comparison(dir, "nominal", "jet_eta", Events, x.Events)
    Comparison(dir, "nominal", "jet_jvt", Events, x.Events)
    Comparison(dir, "nominal", "jet_truthflav", Events, x.Events)
    Comparison(dir, "nominal", "jet_truthPartonLabel", Events, x.Events)
    Comparison(dir, "nominal", "jet_isTrueHS", Events, x.Events)
    Comparison(dir, "nominal", "jet_truthflavExtended", Events, x.Events)
    Comparison(dir, "nominal", "jet_isbtagged_DL1r_77", Events, x.Events)
    Comparison(dir, "nominal", "jet_isbtagged_DL1r_70", Events, x.Events)
    Comparison(dir, "nominal", "jet_isbtagged_DL1r_60", Events, x.Events)
    Comparison(dir, "nominal", "jet_isbtagged_DL1r_85", Events, x.Events)
    Comparison(dir, "nominal", "jet_DL1r", Events, x.Events)
    Comparison(dir, "nominal", "jet_DL1r_pb", Events, x.Events)
    Comparison(dir, "nominal", "jet_DL1r_pc", Events, x.Events)
    Comparison(dir, "nominal", "jet_DL1r_pu", Events, x.Events)
    Comparison(dir, "nominal", "jet_isbtagged_DL1_77", Events, x.Events)
    Comparison(dir, "nominal", "jet_isbtagged_DL1_70", Events, x.Events)
    Comparison(dir, "nominal", "jet_isbtagged_DL1_60", Events, x.Events)
    Comparison(dir, "nominal", "jet_isbtagged_DL1_85", Events, x.Events)
    Comparison(dir, "nominal", "jet_DL1", Events, x.Events)
    Comparison(dir, "nominal", "jet_DL1_pb", Events, x.Events)
    Comparison(dir, "nominal", "jet_DL1_pc", Events, x.Events)
    Comparison(dir, "nominal", "jet_DL1_pu", Events, x.Events)

    #Event 
    Comparison(dir, "nominal", "met_met", Events, x.Events)
    Comparison(dir, "nominal", "met_phi", Events, x.Events)
    Comparison(dir, "nominal", "eventNumber", Events, x.Events)
    Comparison(dir, "nominal", "runNumber", Events, x.Events)
    Comparison(dir, "nominal", "mu", Events, x.Events)
    Comparison(dir, "nominal", "mu_actual", Events, x.Events)

    #Truth Jets
    Comparison(dir, "nominal", "truthjet_pt", Events, x.Events)
    Comparison(dir, "nominal", "truthjet_e", Events, x.Events)
    Comparison(dir, "nominal", "truthjet_phi", Events, x.Events)
    Comparison(dir, "nominal", "truthjet_eta", Events, x.Events)
    Comparison(dir, "nominal", "truthjet_flavour", Events, x.Events)
    Comparison(dir, "nominal", "truthjet_flavour_extended", Events, x.Events)
    Comparison(dir, "nominal", "truthjet_nCHad", Events, x.Events)
    Comparison(dir, "nominal", "truthjet_nBHad", Events, x.Events)

    # Truth Tops
    Comparison(dir, "nominal", "truth_top_pt", Events, x.Events)
    Comparison(dir, "nominal", "truth_top_e", Events, x.Events)
    Comparison(dir, "nominal", "truth_top_phi", Events, x.Events)
    Comparison(dir, "nominal", "truth_top_eta", Events, x.Events)
    Comparison(dir, "nominal", "truth_top_charge", Events, x.Events)
    Comparison(dir, "nominal", "top_FromRes", Events, x.Events)
    
    #Child 
    Comparison(dir, "nominal", "truth_top_child_pt", Events, x.Events)
    Comparison(dir, "nominal", "truth_top_child_e", Events, x.Events)
    Comparison(dir, "nominal", "truth_top_child_phi", Events, x.Events)
    Comparison(dir, "nominal", "truth_top_child_eta", Events, x.Events)
    Comparison(dir, "nominal", "truth_top_child_pdgid", Events, x.Events)

    #Child  init
    Comparison(dir, "nominal", "truth_top_initialState_child_pt", Events, x.Events)
    Comparison(dir, "nominal", "truth_top_initialState_child_e", Events, x.Events)
    Comparison(dir, "nominal", "truth_top_initialState_child_phi", Events, x.Events)
    Comparison(dir, "nominal", "truth_top_initialState_child_eta", Events, x.Events)
    Comparison(dir, "nominal", "top_initialState_child_pdgid", Events, x.Events)

def TestEvent():
    x = EventGenerator(dir, DebugThresh = 10)
    x.SpawnEvents()
