# Event Generator closure test
from Functions.Event.EventGenerator import EventGenerator
import uproot
from Functions.Particles.Particles import *
from Functions.Event.Implementations.Event import Event
from Functions.IO.IO import PickleObject, UnpickleObject
import importlib, inspect


def TestEvents(di):
    x = EventGenerator(di, Start = 0, Stop = 100)
    x.SpawnEvents()
    x.CompileEvent(SingleThread = True)
    return True

def Comparison(di, Tree, Branch, EventContainer, NEvents = -1):
    
    F = uproot.open(di)
    ev = F[Tree + "/" + Branch].array(library = "np")
    
    if NEvents == -1:
        NEvents = len(ev)
   
    Attribute = ""
    Key = ""
    test_e = EventContainer[0][Tree]
    if Branch in test_e.KeyMap:
        Attribute = test_e.KeyMap[Branch]
        Key = "Event"
    else:
        for i in test_e.Objects:
            obj = test_e.Objects[i]
            if Branch in obj.KeyMap:
                Attribute = obj.KeyMap[Branch]
                Key = i
                break

    for i in range(NEvents):
        e_up = ev[i]
        e_my = EventContainer[i][Tree]

        if Key == "Event":
            Value_My = [float(getattr(e_my, Attribute))]
            e_up = [float(e_up)]
        else:
            Object_List = getattr(e_my, Key)
            try:
                Value_My = [float(getattr(k, Attribute)) for k in Object_List]
            except:
                Value_My = [float(l) for k in Object_List for l in getattr(k, Attribute)]

            try:
                e_up = [float(k) for k in e_up]
            except:
                e_up = [float(k) for p in e_up for k in p]
       
        if len(e_up) != len(Value_My):
            return "FAILURE::TREE -> " + Tree + " | Branch -> " + Branch + " Uproot ->" + str(e_up) + " Loader -> " + str(Value_My)

        for k, p in zip(e_up, Value_My):
            try: 
                assert k == p
            except:
                return "FAILURE::TREE -> " + Tree + " | Branch -> " + Branch + " Uproot ->" + str(e_up) + " Loader -> " + str(Value_My)

    return "PASSED::TREE -> " + Tree + " | Branch -> " + Branch



def TestParticleAssignment(di):
    x = EventGenerator(di)
    x.SpawnEvents()
    x.CompileEvent(SingleThread = True, ClearVal = False)

    Events = -1
    TreeTest = "nominal"


    Tests = [
            "el_pt", 
            "el_e", 
            "el_pt", 
            "el_eta", 
            "el_charge", 
            "el_topoetcone20", 
            "el_ptvarcone20", 
            "el_CF", 
            "el_d0sig", 
            "el_delta_z0_sintheta",
            "el_true_type", 
            "el_true_origin", 
            "el_true_firstEgMotherTruthType", 
            "el_true_firstEgMotherTruthOrigin", 
            "el_true_firstEgMotherPdgId", 
            "el_true_IFFclass", 
            "el_true_isPrompt", 
            "el_true_isChargeFl",

            "mu_pt", 
            "mu_e", 
            "mu_pt", 
            "mu_eta", 
            "mu_charge", 
            "mu_topoetcone20", 
            "mu_ptvarcone30", 
            "mu_d0sig", 
            "mu_delta_z0_sintheta",
            "mu_true_type", 
            "mu_true_origin", 
            "mu_true_IFFclass", 
            "mu_true_isPrompt",

            "jet_pt",
            "jet_e",
            "jet_phi",
            "jet_eta",
            "jet_jvt",
            "jet_truthflav",
            "jet_truthPartonLabel",
            "jet_isTrueHS",
            "jet_truthflavExtended",
            "jet_isbtagged_DL1r_77",
            "jet_isbtagged_DL1r_70",
            "jet_isbtagged_DL1r_60",
            "jet_isbtagged_DL1r_85",
            "jet_DL1r",
            "jet_DL1r_pb",
            "jet_DL1r_pc",
            "jet_DL1r_pu",
            "jet_isbtagged_DL1_77",
            "jet_isbtagged_DL1_70",
            "jet_isbtagged_DL1_60",
            "jet_isbtagged_DL1_85",
            "jet_DL1",
            "jet_DL1_pb",
            "jet_DL1_pc",
            "jet_DL1_pu", 
            "jet_map_Ghost", 
            "jet_map_Gtops",

            "met_met",
            "met_phi",
            "eventNumber",
            "runNumber",
            "mu",
            "mu_actual",

            "truthjet_pt",
            "truthjet_e",
            "truthjet_phi",
            "truthjet_eta",
            "truthjet_pdgid", 
            "GhostTruthJetMap",

            "truth_top_pt",
            "truth_top_e",
            "truth_top_phi",
            "truth_top_eta",
            "truth_top_FromRes",
   
            "topPreFSR_pt",
            "topPreFSR_e",
            "topPreFSR_phi",
            "topPreFSR_eta",
            "topPreFSR_charge",
            "topPreFSR_status",

            "topPostFSR_pt",
            "topPostFSR_e",
            "topPostFSR_phi",
            "topPostFSR_eta",
            "topPostFSR_charge",
            "Gtop_FromRes",

            "truth_top_child_pt",
            "truth_top_child_e",
            "truth_top_child_phi",
            "truth_top_child_eta", 
            "truth_top_child_charge",
            "truth_top_child_pdgid",
            "topPostFSRchildren_pt",
            "topPostFSRchildren_e",
            "topPostFSRchildren_phi",
            "topPostFSRchildren_eta",
            "topPostFSRchildren_charge",
            "topPostFSRchildren_pdgid"]
    
    for i in Tests:
        print(Comparison(di, TreeTest, i, x.Events, Events))
    return True


def TestSignalMultipleFile(di):
    
    ev = EventGenerator(di, Stop = 1000)
    ev.SpawnEvents()
    ev.CompileEvent(SingleThread = True)
    
    for i in ev.Events:
        if i == 1000-1:
            print(i, ev.Events[i], ev.EventIndexFileLookup(i)) 
            return True

def TestSignalDirectory(di):
    
    ev = EventGenerator(di, Stop = 1000)
    ev.SpawnEvents()
    ev.CompileEvent(SingleThread = True)
   
    c = 0
    Passed = False
    for i in ev.Events:
        if c == 1000-1:
            print(i, ev.Events[i], ev.EventIndexFileLookup(i)) 
            Passed = True
            c = 0
        c+=1
    return Passed

