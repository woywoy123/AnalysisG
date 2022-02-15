# Event Generator closure test
from Functions.Event.EventGenerator import EventGenerator
import uproot
from Functions.Particles.Particles import *
from Functions.Event.Event import Event
from Functions.IO.IO import PickleObject, UnpickleObject


dir = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root"
def TestEvents():
    x = EventGenerator(dir, Start = 0, Stop = 100)
    x.SpawnEvents()
    x.CompileEvent(SingleThread = True)
    return True

def Comparison(dir, Tree, Branch, Events, Spawned):
    el = Electron()
    mu = Muon()
    truthjet = TruthJet()
    jet = Jet()
    top = Top()
    child = Truth_Top_Child()
    child_init = Truth_Top_Child_Init()
    rcsubjet = RCSubJet()
    rcjet = RCJet()

    event = Event()
    
    try:
        F = uproot.open(dir)
        en = F[Tree + "/" + Branch].array(library = "np")
    except:
        print("SKIPPED::" + Branch + " " + Tree + " ")
        return 

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
        
        it = 0
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

            if Branch in rcjet.KeyMap:
                Compare = Ev.RCJets[x].GetAttributeFromKeyMap(Branch)
            
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
                        Compare = Ev.TruthChildren[it].GetAttributeFromKeyMap(Branch)
                        it += 1
                    if Branch in child_init.KeyMap:
                        Compare = Ev.TruthChildren_init[it].GetAttributeFromKeyMap(Branch)
                        it += 1
                    if Branch in rcsubjet.KeyMap:
                        Compare = Ev.RCSubJets[it].GetAttributeFromKeyMap(Branch)
                        it += 1

                    if Compare != f[x][k]:
                        print("FAILURE::" + Branch + " " + Tree + " " + str(x) + " " + str(k) + " " + str(Compare))
                        assert Compare == f[x][k]
                    comp = True

            if comp:
                pass
            else:
                print("FAILURE::" + Branch + " " + Tree + " " + str(x) )

    print("PASSED:: " + Branch + " " + Tree)


def TestParticleAssignment():
    Events = -1
    x = EventGenerator(dir)
    x.SpawnEvents()
    x.CompileEvent(SingleThread = True, ClearVal = False)
    TreeTest = "nominal"

    #Electrons 
    Comparison(dir, TreeTest, "el_pt", Events, x.Events)
    Comparison(dir, TreeTest, "el_e", Events, x.Events)
    Comparison(dir, TreeTest, "el_phi", Events, x.Events)
    Comparison(dir, TreeTest, "el_eta", Events, x.Events)
    Comparison(dir, TreeTest, "el_charge", Events, x.Events)
    Comparison(dir, TreeTest, "el_topoetcone20", Events, x.Events)
    Comparison(dir, TreeTest, "el_ptvarcone20", Events, x.Events)
    Comparison(dir, TreeTest, "el_CF", Events, x.Events)
    Comparison(dir, TreeTest, "el_d0sig", Events, x.Events)
    Comparison(dir, TreeTest, "el_delta_z0_sintheta", Events, x.Events)
    Comparison(dir, TreeTest, "el_true_type", Events, x.Events)
    Comparison(dir, TreeTest, "el_true_origin", Events, x.Events)
    Comparison(dir, TreeTest, "el_true_firstEgMotherTruthType", Events, x.Events)
    Comparison(dir, TreeTest, "el_true_firstEgMotherTruthOrigin", Events, x.Events)
    Comparison(dir, TreeTest, "el_true_firstEgMotherPdgId", Events, x.Events)
    Comparison(dir, TreeTest, "el_true_IFFclass", Events, x.Events)
    Comparison(dir, TreeTest, "el_true_isPrompt", Events, x.Events)
    Comparison(dir, TreeTest, "el_true_isChargeFl", Events, x.Events)
    
    #Muons
    Comparison(dir, TreeTest, "mu_pt", Events, x.Events)
    Comparison(dir, TreeTest, "mu_e", Events, x.Events)
    Comparison(dir, TreeTest, "mu_phi", Events, x.Events)
    Comparison(dir, TreeTest, "mu_eta", Events, x.Events)
    Comparison(dir, TreeTest, "mu_charge", Events, x.Events)
    Comparison(dir, TreeTest, "mu_topoetcone20", Events, x.Events)
    Comparison(dir, TreeTest, "mu_ptvarcone30", Events, x.Events)
    Comparison(dir, TreeTest, "mu_d0sig", Events, x.Events)
    Comparison(dir, TreeTest, "mu_delta_z0_sintheta", Events, x.Events)
    Comparison(dir, TreeTest, "mu_true_type", Events, x.Events)
    Comparison(dir, TreeTest, "mu_true_origin", Events, x.Events)
    Comparison(dir, TreeTest, "mu_true_IFFclass", Events, x.Events)
    Comparison(dir, TreeTest, "mu_true_isPrompt", Events, x.Events)

    #jets
    Comparison(dir, TreeTest, "jet_pt", Events, x.Events)
    Comparison(dir, TreeTest, "jet_e", Events, x.Events)
    Comparison(dir, TreeTest, "jet_phi", Events, x.Events)
    Comparison(dir, TreeTest, "jet_eta", Events, x.Events)
    Comparison(dir, TreeTest, "jet_jvt", Events, x.Events)
    Comparison(dir, TreeTest, "jet_truthflav", Events, x.Events)
    Comparison(dir, TreeTest, "jet_truthPartonLabel", Events, x.Events)
    Comparison(dir, TreeTest, "jet_isTrueHS", Events, x.Events)
    Comparison(dir, TreeTest, "jet_truthflavExtended", Events, x.Events)
    Comparison(dir, TreeTest, "jet_isbtagged_DL1r_77", Events, x.Events)
    Comparison(dir, TreeTest, "jet_isbtagged_DL1r_70", Events, x.Events)
    Comparison(dir, TreeTest, "jet_isbtagged_DL1r_60", Events, x.Events)
    Comparison(dir, TreeTest, "jet_isbtagged_DL1r_85", Events, x.Events)
    Comparison(dir, TreeTest, "jet_DL1r", Events, x.Events)
    Comparison(dir, TreeTest, "jet_DL1r_pb", Events, x.Events)
    Comparison(dir, TreeTest, "jet_DL1r_pc", Events, x.Events)
    Comparison(dir, TreeTest, "jet_DL1r_pu", Events, x.Events)
    Comparison(dir, TreeTest, "jet_isbtagged_DL1_77", Events, x.Events)
    Comparison(dir, TreeTest, "jet_isbtagged_DL1_70", Events, x.Events)
    Comparison(dir, TreeTest, "jet_isbtagged_DL1_60", Events, x.Events)
    Comparison(dir, TreeTest, "jet_isbtagged_DL1_85", Events, x.Events)
    Comparison(dir, TreeTest, "jet_DL1", Events, x.Events)
    Comparison(dir, TreeTest, "jet_DL1_pb", Events, x.Events)
    Comparison(dir, TreeTest, "jet_DL1_pc", Events, x.Events)
    Comparison(dir, TreeTest, "jet_DL1_pu", Events, x.Events)

    #Event 
    Comparison(dir, TreeTest, "met_met", Events, x.Events)
    Comparison(dir, TreeTest, "met_phi", Events, x.Events)
    Comparison(dir, TreeTest, "eventNumber", Events, x.Events)
    Comparison(dir, TreeTest, "runNumber", Events, x.Events)
    Comparison(dir, TreeTest, "mu", Events, x.Events)
    Comparison(dir, TreeTest, "mu_actual", Events, x.Events)

    #Truth Jets
    Comparison(dir, TreeTest, "truthjet_pt", Events, x.Events)
    Comparison(dir, TreeTest, "truthjet_e", Events, x.Events)
    Comparison(dir, TreeTest, "truthjet_phi", Events, x.Events)
    Comparison(dir, TreeTest, "truthjet_eta", Events, x.Events)
    Comparison(dir, TreeTest, "truthjet_flavour", Events, x.Events)
    Comparison(dir, TreeTest, "truthjet_flavour_extended", Events, x.Events)
    Comparison(dir, TreeTest, "truthjet_nCHad", Events, x.Events)
    Comparison(dir, TreeTest, "truthjet_nBHad", Events, x.Events)

    # Truth Tops
    Comparison(dir, TreeTest, "truth_top_pt", Events, x.Events)
    Comparison(dir, TreeTest, "truth_top_e", Events, x.Events)
    Comparison(dir, TreeTest, "truth_top_phi", Events, x.Events)
    Comparison(dir, TreeTest, "truth_top_eta", Events, x.Events)
    Comparison(dir, TreeTest, "truth_top_charge", Events, x.Events)
    Comparison(dir, TreeTest, "top_FromRes", Events, x.Events)
    
    #Child 
    Comparison(dir, TreeTest, "truth_top_child_pt", Events, x.Events)
    Comparison(dir, TreeTest, "truth_top_child_e", Events, x.Events)
    Comparison(dir, TreeTest, "truth_top_child_phi", Events, x.Events)
    Comparison(dir, TreeTest, "truth_top_child_eta", Events, x.Events) 
    Comparison(dir, TreeTest, "truth_top_child_pdgid", Events, x.Events)

    #Child  init
    Comparison(dir, TreeTest, "truth_top_initialState_child_pt", Events, x.Events)
    Comparison(dir, TreeTest, "truth_top_initialState_child_e", Events, x.Events)
    Comparison(dir, TreeTest, "truth_top_initialState_child_phi", Events, x.Events)
    Comparison(dir, TreeTest, "truth_top_initialState_child_eta", Events, x.Events)
    Comparison(dir, TreeTest, "top_initialState_child_pdgid", Events, x.Events)
    
    #RCJets 
    Comparison(dir, TreeTest, "rcjet_pt", Events, x.Events)
    Comparison(dir, TreeTest, "rcjet_e", Events, x.Events)
    Comparison(dir, TreeTest, "rcjet_phi", Events, x.Events)
    Comparison(dir, TreeTest, "rcjet_eta", Events, x.Events)
    Comparison(dir, TreeTest, "rcjet_d12", Events, x.Events)
    Comparison(dir, TreeTest, "rcjet_d23", Events, x.Events)

    #RCJets Sub
    Comparison(dir, TreeTest, "rcjetsub_pt", Events, x.Events)
    Comparison(dir, TreeTest, "rcjetsub_e", Events, x.Events)
    Comparison(dir, TreeTest, "rcjetsub_phi", Events, x.Events)
    Comparison(dir, TreeTest, "rcjetsub_eta", Events, x.Events) 
    
    return True


def TestSignalMultipleFile():
    dir = "/CERN/Grid/SignalSamples/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/"
    
    ev = EventGenerator(dir, Stop = 1000)
    ev.SpawnEvents()
    ev.CompileEvent(SingleThread = True)
    
    for i in ev.Events:
        if i == 1000:
            print(i, ev.Events[i], ev.EventIndexFileLookup(i)) 
            return True

def TestSignalDirectory():
    dir = "/CERN/Grid/SignalSamples/"
    
    ev = EventGenerator(dir, Stop = 1000)
    ev.SpawnEvents()
    ev.CompileEvent(SingleThread = True)
   
    c = 0
    Passed = False
    for i in ev.Events:
        if c == 1000:
            print(i, ev.Events[i], ev.EventIndexFileLookup(i)) 
            Passed = True
            c = 0
        c+=1
    return Passed

