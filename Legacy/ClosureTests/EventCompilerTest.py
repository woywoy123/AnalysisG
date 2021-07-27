from BaseFunctions.IO import *
from BaseFunctions.EventsManager import *
from BaseFunctions.VariableManager import *

def TestObjectVariableAssignment(): 

    def Attest(var1, var2, Attr, index):
        
        Passed = False
        v2 = getattr(var2, Attr)
        if len(v2[index]) == len(var1):
            Passed = True

        for k in range(len(v2[index])):
            va1 = round(float(getattr(var1[k], Attr)), 4)
            va2 = round(float(v2[index][k]), 4)
            if  va1 == va2:
                Passed = True
            else:
                Passed = False
                print("=====> ", va1, va2, type(va1), type(va2),  Attr, index)

            if Passed == False:
                break
        return Passed

    files = "/CERN/Grid/SignalSamples/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root"

    V = ["topoetcone20", "ptvarcone20", "CF", "d0sig", "delta_z0_sintheta", "true_type", "true_origin", "true_firstEgMotherTruthType", "true_firstEgMotherTruthOrigin", "true_firstEgMotherPdgId", "true_IFFclass", "true_isPrompt", "true_isChargeFl"]
    electron_branch = ["el_pt", "el_eta", "el_phi", "el_e", "el_charge", "el_topoetcone20", "el_ptvarcone20", "el_CF", "el_d0sig", "el_delta_z0_sintheta", "el_true_type", "el_true_origin", "el_true_firstEgMotherTruthType", "el_true_firstEgMotherTruthOrigin", "el_true_firstEgMotherPdgId", "el_true_IFFclass", "el_true_isPrompt", "el_true_isChargeFl"]

    # Test if the Branches are in the files:
    b = BranchVariable(files, "nominal", electron_branch)
    ev = EventCompiler(files, "nominal", electron_branch)
    ev_d = ev.EventDictionary

    for i in range(len(b.EventNumber)):
        nr = b.EventNumber[i]
        p_list = ev_d[nr]
       
        assert(Attest(p_list, b, "Phi", i))
        assert(Attest(p_list, b, "E", i))
        assert(Attest(p_list, b, "Pt", i))
        assert(Attest(p_list, b, "Eta", i))
        assert(Attest(p_list, b, "Charge", i))
        
        for k in V:
            assert(Attest(p_list, b, k, i))

    print("ELECTRON::PASSED")

    V = ["topoetcone20", "ptvarcone30", "d0sig", "delta_z0_sintheta", "true_type", "true_origin", "true_IFFclass", "true_isPrompt"]    
    muon_branch = ["mu_pt", "mu_eta", "mu_phi", "mu_e", "mu_charge", "mu_topoetcone20", "mu_ptvarcone30", "mu_d0sig", "mu_delta_z0_sintheta", "mu_true_type", "mu_true_origin", "mu_true_IFFclass", "mu_true_isPrompt"]

    # Test if the Branches are in the files:
    b = BranchVariable(files, "nominal", muon_branch)
    ev = EventCompiler(files, "nominal", muon_branch).EventDictionary

    for i in range(len(b.EventNumber)):
        nr = b.EventNumber[i]
        p_list = ev[nr]
       
        assert(Attest(p_list, b, "Phi", i))
        assert(Attest(p_list, b, "E", i))
        assert(Attest(p_list, b, "Pt", i))
        assert(Attest(p_list, b, "Eta", i))
        assert(Attest(p_list, b, "Charge", i))
        for k in V:
            assert(Attest(p_list, b, k, i))

    print("MUON::PASSED")

    V = ["jvt", "truthflav", "truthPartonLabel", "isTrueHS", "DL1_77", "DL1_70", "DL1_60", "DL1_85", "DL1", "DL1_pb", "DL1_pc", "DL1_pu", "DL1r_77", "DL1r_70", "DL1r_60", "DL1r_85", "DL1r", "DL1r_pb", "DL1r_pc", "DL1r_pu"]    

    jet_branch = ["jet_pt", "jet_eta", "jet_phi", "jet_e", "jet_jvt", "jet_truthflav", "jet_truthPartonLabel", "jet_isTrueHS", "jet_isbtagged_DL1_77", "jet_isbtagged_DL1_70", "jet_isbtagged_DL1_60", "jet_isbtagged_DL1_85", "jet_DL1", "jet_DL1_pb", "jet_DL1_pc", "jet_DL1_pu", "jet_isbtagged_DL1r_77", "jet_isbtagged_DL1r_70", "jet_isbtagged_DL1r_60", "jet_isbtagged_DL1r_85", "jet_DL1r", "jet_DL1r_pb", "jet_DL1r_pc", "jet_DL1r_pu"]  

    # Test if the Branches are in the files:
    b = BranchVariable(files, "nominal", jet_branch)
    ev = EventCompiler(files, "nominal", jet_branch)
    ev_d = ev.EventDictionary

    for i in range(len(b.EventNumber)):
        nr = b.EventNumber[i]
        p_list = ev_d[nr]
       
        assert(Attest(p_list, b, "Phi", i))
        assert(Attest(p_list, b, "E", i))
        assert(Attest(p_list, b, "Pt", i))
        assert(Attest(p_list, b, "Eta", i))
        for k in V:
            assert(Attest(p_list, b, k, i))       
 
    print("JET::PASSED")


    V = ["Flavour", "nChad", "nBhad"]    
    truthjet_branch = ["truthjet_pt", "truthjet_eta", "truthjet_phi", "truthjet_e", "truthjet_flavour", "truthjet_nCHad", "truthjet_nBHad"]  

    # Test if the Branches are in the files:
    b = BranchVariable(files, "nominal", truthjet_branch)
    ev = EventCompiler(files, "nominal", truthjet_branch)
    ev_d = ev.EventDictionary

    for i in range(len(b.EventNumber)):
        nr = b.EventNumber[i]
        p_list = ev_d[nr]
       
        assert(Attest(p_list, b, "Phi", i))
        assert(Attest(p_list, b, "E", i))
        assert(Attest(p_list, b, "Pt", i))
        assert(Attest(p_list, b, "Eta", i))

        for k in V:
            assert(Attest(p_list, b, k, i))       
    print("TRUTH JET::PASSED")

    truthtop_branch = ["truth_top_pt", "truth_top_eta", "truth_top_phi", "truth_top_e", "truth_top_charge", "top_FromRes"]  
    b = BranchVariable(files, "nominal", truthtop_branch)
    ev = EventCompiler(files, "nominal", truthtop_branch)
    ev_d = ev.EventDictionary

    for i in range(len(b.EventNumber)):
        nr = b.EventNumber[i]
        p_list = ev_d[nr]
       
        assert(Attest(p_list, b, "Phi", i))
        assert(Attest(p_list, b, "E", i))
        assert(Attest(p_list, b, "Pt", i))
        assert(Attest(p_list, b, "Eta", i))
        assert(Attest(p_list, b, "Charge", i))
            
        for j in range(len(b.IsSignal[i])):
            assert(b.IsSignal[i][j] == p_list[j].IsSignal)
    print("TRUTH TOP::PASSED")
    
    init_c = "truth_top_initialState_child"
    truthtop_branch = ["top_FromRes", init_c+"_pt", init_c+"_eta", init_c+"_phi", init_c+"_e", "top_initialState_child_pdgid"] 

    b = BranchVariable(files, "nominal", truthtop_branch)
    ev = TruthCompiler(files).EventDictionary

    for i in range(len(b.EventNumber)):
        nr = b.EventNumber[i]
        p_list = ev[nr].TruthMatch
    
        for k in range(len(b.IsSignal[i])):
            p = p_list[k]
            assert(p.IsSignal == b.IsSignal[i][k])
            
            for pc in range(len(b.Eta[i][k])):
                px = p.init_DecayProducts[pc] 

                assert(round(float(px.Eta), 5) == round(float(b.Eta[i][k][pc]), 5))
                assert(round(float(px.Pt), 5) == round(float(b.Pt[i][k][pc]), 5))
                assert(round(float(px.Phi), 5) == round(float(b.Phi[i][k][pc]), 5))
                assert(round(float(px.E), 5) == round(float(b.E[i][k][pc]), 5))

    print("TRUTH WITH CHILD MATCHING::PASSED")

def VisualizeParticleMatching():
    from BaseFunctions.Plotting import PlotADetector
    from BaseFunctions.IO import PickleObject, UnpickleObject

    #files = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root"
    #TC = TruthCompiler(files, Debug = True)
    #PickleObject(TC.EventDictionary, "TruthCompiler")

    te = UnpickleObject("TruthCompiler")
    
    for i in te:
        
        print("===============")

        #assert me[i].EventNumber == te[i].EventNumber
        E = te[i]
        
        
        l = 0 
        for k in E.RCJets:
            l += len(k.Sub_Jets)

        print(len(E.TruthJets), len(E.Jets), len(E.RCJets), l, len(E.Leptons))
        if len(E.TruthJets) < len(E.Jets):
            print("!!!!!!!!!!")


        dct = {}
        for x in E.Jets:
            for j in E.TruthJets:
                dct[j.KinematicDifference(x)] = [j, x]


        Used = []
        for p in sorted(dct.items()):
            tj = p[1][0]
            j = p[1][1]
            if j not in Used:
                tj.Sub_Jets.append(j)
                Used.append(j)
                print(p[0], j.Index, tj.Index, [j.truthflavExtended, j.truthflav], [tj.Flavour, tj.Extended] )
       
        dct.clear()
        Used.clear()
        for p in E.RCJets:
            for k in p.Sub_Jets:
                for j in E.Jets:
                    dct[j.KinematicDifference(k)] = [k, j]
        
        for p in sorted(dct.items()):
            if p[1][0] not in Used:
                p[1][1].Sub_Jets.append(p[1][0])
                Used.append(p[1][0])

        #for p in E.TruthJets:
        #    print(p.Type)
        #    for g in p.Sub_Jets:
        #        print(g.Type)
        #        for k in g.Sub_Jets:
        #            print(k.Type)
 



        
        
        #PlotADetector(te[i].DetectorParticles, te[i].TruthMatch)



