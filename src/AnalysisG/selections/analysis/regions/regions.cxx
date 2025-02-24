#include "regions.h"

regions::regions(){this -> name = "regions";}
regions::~regions(){}

selection_template* regions::clone(){
    return (selection_template*)new regions();
}

void regions::merge(selection_template* sl){
    regions* slt = (regions*)sl; 
    merge_data(&this -> output, &slt -> output); 
}

bool regions::selection(event_template* ev){
    return true;
}

bool regions::strategy(event_template* ev){
    ssml_mc20* evn = (ssml_mc20*)ev; 
    std::vector<jet*> jets; 
    std::vector<lepton*> leps; 
    std::vector<electron*> elx; 
    this -> upcast(&evn -> Leptsn, &leps);  
    this -> upcast(&evn -> Electrons, &elx); 
    this -> upcast(&evn -> Jets, &jets); 

    float event_weight = evn -> weight_mc * evn -> weight_pileup * evn -> weight_beamspot;
    event_weight *= evn -> weight_jvt_effSF * evn -> weight_lep_tightSF * evn -> weight_ftag_effSF; 

    std::string camp = this -> meta_data -> campaign; 
    double sow   = this -> meta_data -> sum_of_weights; 
    double cross = this -> meta_data -> cross_section_pb; 
    if      (camp == "mc20a"){event_weight *= 36646.74;}
    else if (camp == "mc20d"){event_weight *= 44630.6;}
    else if (camp == "mc20e"){event_weight *= 58791.6;}
    event_weight *= cross/sow; 
    
    // ----- nBjets_GN2v01_85WP_NOSYS ----- //
    int nbjet_85wp = 0; 
    for (size_t x(0); x < jets.size(); ++x){nbjet_85wp += (jets[x] -> sel_85 == 1);}

    // ----- pass_SSee_passECIDS_NOSYS ----- //
    bool pass_ssee = evn -> pass_ssee == 1;
    if (evn -> Electrons.size() > 1){pass_ssee *= elx[0] -> pass_ecids == 1 && elx[1] -> pass_ecids == 1;}
    else {pass_ssee *= false;}

    // ----- pass_SSem_passECIDS_NOSYS ----- //
    bool pass_ssem = evn -> pass_ssem == 1;
    if (evn -> Electrons.size() > 0){pass_ssem *= elx[0] -> pass_ecids == 1;}
    else {pass_ssem *= false;}
    bool pass_ssmm = evn -> pass_ssmm == 1; 

    // ----- pass_SSem_passEtaCut_NOSYS ----- //
    bool pass_ssem_cut = evn -> pass_ssem == 1 ; 
    if (evn -> Electrons.size() > 0){pass_ssem_cut *= std::abs(elx[0] -> eta) < 1.37;}
    else {pass_ssem_cut *= false;}


    // ----- HT_all_NOSYS ----- //
    float ht_all = evn -> HT_all; 
    float ht_all_gev = evn -> HT_all / 1000.0; 
    float met_gev = evn -> met / 1000.0; 

    bool  lepton_0_pt = false; 
    float lepton_2_pt = 0; 
    float lep_sum_pt = 0; 
    float lepton_0_charge = 0; 
    float lepton_1_charge = 0; 

    float sum_lepton_3_charge = 0; 
    int lepton_0_ambiguity = 99; 
    int lepton_1_ambiguity = 99; 
    int lepton_2_ambiguity = 99; 
    if (leps.size() > 0){
        lepton_0_charge = leps[0] -> charge; 
        lepton_0_ambiguity = leps[0] -> ambiguity; 
        lepton_0_pt = (leps[0] -> pt / 1000.0) > 28.0;
    }

    if (leps.size() > 1){
        lepton_1_charge = leps[1] -> charge; 
        lepton_1_ambiguity = leps[1] -> ambiguity; 
        lep_sum_pt = (leps[0] -> pt + leps[1] -> pt)/1000.0; 
    }

    if (leps.size() > 2){
        sum_lepton_3_charge = leps[0] -> charge + leps[1] -> charge + leps[2] -> charge;
        lepton_2_ambiguity = leps[2] -> ambiguity; 
        lepton_2_pt = leps[2] -> pt / 1000.0; 
    }

    package_t data = package_t(); 

    data.CRttbarCO2l_CO.variable1 = 1; 
    data.CRttbarCO2l_CO.weight = event_weight; 
    data.CRttbarCO2l_CO.passed *= (lepton_0_pt && (pass_ssee || pass_ssem));
    data.CRttbarCO2l_CO.passed *= (lepton_0_ambiguity == 2 || lepton_1_ambiguity == 2); 
    data.CRttbarCO2l_CO.passed *= (nbjet_85wp >= 1) && (evn -> n_jets >= 4) && (evn -> n_jets < 6); 
    // lepton_0_pt_GeV_NOSYS>28 
    // && (pass_SSee_passECIDS_NOSYS||pass_SSem_passECIDS_NOSYS) 
    // && (lepton_0_DFCommonAddAmbiguity_NOSYS==2||lepton_1_DFCommonAddAmbiguity_NOSYS==2) 
    // && nBjets_GN2v01_85WP_NOSYS>=1 
    // && nJets>=4 && nJets<6
    // name: "1"
    // definition: "zero_NOSYS"
    // min: 0
    // max: 1
    // number_of_bins: 1


    data.CRttbarCO2l_CO_2b.variable1 = 1; 
    data.CRttbarCO2l_CO_2b.weight = event_weight; 
    data.CRttbarCO2l_CO_2b.passed *= lepton_0_pt * (pass_ssee || pass_ssem); 
    data.CRttbarCO2l_CO_2b.passed *= (lepton_0_ambiguity == 2 || lepton_1_ambiguity == 2); 
    data.CRttbarCO2l_CO_2b.passed *= (nbjet_85wp >= 2) * (evn -> n_jets >= 4) * (evn -> n_jets < 6); 
    // lepton_0_pt_GeV_NOSYS>28 && (pass_SSee_passECIDS_NOSYS||pass_SSem_passECIDS_NOSYS) 
    // && (lepton_0_DFCommonAddAmbiguity_NOSYS==2||lepton_1_DFCommonAddAmbiguity_NOSYS==2) 
    // && nBjets_GN2v01_85WP_NOSYS>=2 && nJets>=4 && nJets<6
    // name: "1"
    // definition: "zero_NOSYS"
    // min: 0
    // max: 1
    // number_of_bins: 1

    data.CRttbarCO2l_gstr.variable1 = 1; 
    data.CRttbarCO2l_gstr.weight = event_weight; 
    data.CRttbarCO2l_gstr.passed *=  lepton_0_pt * (pass_ssee || pass_ssem); 
    data.CRttbarCO2l_gstr.passed *=  (lepton_0_ambiguity == 1 || lepton_1_ambiguity == 1); 
    data.CRttbarCO2l_gstr.passed *= !(lepton_0_ambiguity == 2 || lepton_1_ambiguity == 2); 
    data.CRttbarCO2l_gstr.passed *=  (nbjet_85wp >= 1) * (evn -> n_jets >= 4) * (evn -> n_jets < 6); 
    // lepton_0_pt_GeV_NOSYS>28 && (pass_SSee_passECIDS_NOSYS||pass_SSem_passECIDS_NOSYS) 
    // && (lepton_0_DFCommonAddAmbiguity_NOSYS==1||lepton_1_DFCommonAddAmbiguity_NOSYS==1)
    // && !(lepton_0_DFCommonAddAmbiguity_NOSYS==2||lepton_1_DFCommonAddAmbiguity_NOSYS==2) 
    // && nBjets_GN2v01_85WP_NOSYS>=1 && nJets>=4 && nJets<6
    // name: "1"
    // definition: "zero_NOSYS"
    // min: 0
    // max: 1
    // number_of_bins: 1


    data.CRttbarCO2l_gstr_2b.variable1 = 1; 
    data.CRttbarCO2l_gstr_2b.weight = event_weight; 
    data.CRttbarCO2l_gstr_2b.passed *=  lepton_0_pt * (pass_ssee || pass_ssem); 
    data.CRttbarCO2l_gstr_2b.passed *=  (lepton_0_ambiguity == 1 || lepton_1_ambiguity == 1); 
    data.CRttbarCO2l_gstr_2b.passed *= !(lepton_0_ambiguity == 2 || lepton_1_ambiguity == 2); 
    data.CRttbarCO2l_gstr_2b.passed *=  (nbjet_85wp >= 2) * (evn -> n_jets >= 4) * (evn -> n_jets < 6); 
    // lepton_0_pt_GeV_NOSYS>28 && (pass_SSee_passECIDS_NOSYS||pass_SSem_passECIDS_NOSYS) 
    // &&  (lepton_0_DFCommonAddAmbiguity_NOSYS==1 || lepton_1_DFCommonAddAmbiguity_NOSYS==1)
    // && !(lepton_0_DFCommonAddAmbiguity_NOSYS==2 || lepton_1_DFCommonAddAmbiguity_NOSYS==2) 
    // && nBjets_GN2v01_85WP_NOSYS>=2 && nJets>=4 && nJets<6"
    // name: "1"
    // definition: "zero_NOSYS"
    // min: 0
    // max: 1
    // number_of_bins: 1


    data.CR1b3lem.variable1 = lepton_2_pt; 
    data.CR1b3lem.weight = event_weight; 
    data.CR1b3lem.passed *= lepton_0_pt && std::abs(sum_lepton_3_charge) == 1; 
    data.CR1b3lem.passed *= (nbjet_85wp == 1 && lepton_0_ambiguity < 0 && lepton_1_ambiguity < 0); 

    bool blk1 = (evn -> pass_eem_zveto || evn -> pass_eee_zveto);
    blk1 *= (ht_all < 275000.0 && ht_all > 100000.0); 
    blk1 *= (met_gev > 35); 

    bool blk2 = (evn -> pass_emm_zveto || evn -> pass_mmm_zveto);
    blk2 *= (ht_all > 100000.0 && ht_all < 300000.0); 
    blk2 *= (lepton_2_ambiguity < 0 && met_gev > 50);  

    // ------ seems like a bug? should maybe be *= for both? ------ //
    data.CR1b3lem.passed *= blk1; 
    data.CR1b3lem.passed += blk2; 
    // lepton_0_pt_GeV_NOSYS>28 
    // && (
    //            (pass_eem_ZVeto_NOSYS||pass_eee_ZVeto_NOSYS) 
    //         && nBjets_GN2v01_85WP_NOSYS==1 
    //         && lepton_0_DFCommonAddAmbiguity_NOSYS<0 
    //         && lepton_1_DFCommonAddAmbiguity_NOSYS<0 
    //         && HT_all_NOSYS>100000 
    //         && HT_all_NOSYS<275000 
    //         && abs(lepton_0_charge_NOSYS+lepton_1_charge_NOSYS+lepton_2_charge_NOSYS)==1 
    //         && met_met_NOSYS/1e3>35
    // ) 
    // || (
    //            (pass_emm_ZVeto_NOSYS||pass_mmm_ZVeto_NOSYS) 
    //         && nBjets_GN2v01_85WP_NOSYS==1 
    //         && HT_all_NOSYS>100000 
    //         && HT_all_NOSYS<300000 
    //         && (
    //                 lepton_0_DFCommonAddAmbiguity_NOSYS<0 
    //              && lepton_1_DFCommonAddAmbiguity_NOSYS<0 
    //              && lepton_2_DFCommonAddAmbiguity_NOSYS<0
    //        ) 
    //        && abs(lepton_0_charge_NOSYS+lepton_1_charge_NOSYS+lepton_2_charge_NOSYS)==1 
    //        && met_met_NOSYS/1e3>50
    // )
    // name: "lep_2_pt"
    // title : "CR1b3lem;lep_2_pt;Y axis title"
    // definition: "lepton_2_pt_GeV_NOSYS"
    // type: "double"
    // bin_edges: [15.,30.,50.]


    data.CR1b3le.variable1 = lepton_2_pt; 
    data.CR1b3le.weight = event_weight; 
    data.CR1b3le.passed *= lepton_0_pt * (evn -> pass_eem_zveto || evn -> pass_eee_zveto);  
    data.CR1b3le.passed *= (nbjet_85wp == 1 && lepton_0_ambiguity < 0 && lepton_1_ambiguity < 0);  
    data.CR1b3le.passed *= (ht_all > 100000.0 && ht_all < 275000.0); 
    data.CR1b3le.passed *= (std::abs(sum_lepton_3_charge) == 1 && met_gev > 35); 
    // lepton_0_pt_GeV_NOSYS>28 
    // && (pass_eem_ZVeto_NOSYS || pass_eee_ZVeto_NOSYS) 
    // && nBjets_GN2v01_85WP_NOSYS == 1 
    // && lepton_0_DFCommonAddAmbiguity_NOSYS < 0 
    // && lepton_1_DFCommonAddAmbiguity_NOSYS < 0 
    // && HT_all_NOSYS > 100000 
    // && HT_all_NOSYS < 275000 
    // && abs(lepton_0_charge_NOSYS+lepton_1_charge_NOSYS+lepton_2_charge_NOSYS)==1 
    // && met_met_NOSYS/1e3>35"
    // name: "lep_2_pt"
    // title : "CR1b3le;lep_2_pt;Y axis title"
    // definition: "lepton_2_pt_GeV_NOSYS"
    // type: "double"
    // bin_edges: [15.,30.,50.]

    data.CR1b3lm.variable1 = lepton_2_pt; 
    data.CR1b3lm.weight = event_weight; 
    data.CR1b3lm.passed *= lepton_0_pt * (evn -> pass_emm_zveto || evn -> pass_mmm_zveto);  
    data.CR1b3lm.passed *= (nbjet_85wp == 1 && ht_all > 100000.0 && ht_all < 300000.0); 
    data.CR1b3lm.passed *= lepton_0_ambiguity < 0 && lepton_1_ambiguity < 0 && lepton_2_ambiguity < 0; 
    data.CR1b3lm.passed *= (std::abs(sum_lepton_3_charge) == 1 && met_gev > 50); 
    // lepton_0_pt_GeV_NOSYS>28 
    // && (pass_emm_ZVeto_NOSYS || pass_mmm_ZVeto_NOSYS) 
    // && nBjets_GN2v01_85WP_NOSYS==1 
    // && HT_all_NOSYS>100000 
    // && HT_all_NOSYS<300000 
    // && (lepton_0_DFCommonAddAmbiguity_NOSYS<0 && lepton_1_DFCommonAddAmbiguity_NOSYS<0 && lepton_2_DFCommonAddAmbiguity_NOSYS<0) 
    // && abs(lepton_0_charge_NOSYS+lepton_1_charge_NOSYS+lepton_2_charge_NOSYS)==1 
    // && met_met_NOSYS/1e3>50"
    // name: "lep_2_pt"
    // definition: "lepton_2_pt_GeV_NOSYS"
    // type: "double"
    // bin_edges: [15.,30.,50.]

    data.CRttW2l_plus.variable1 = evn -> n_jets; 
    data.CRttW2l_plus.weight = event_weight; 
    data.CRttW2l_plus.passed *= lepton_0_pt * ((pass_ssem && pass_ssem_cut) || evn -> pass_ssmm);  
    data.CRttW2l_plus.passed *= (evn -> n_jets >= 4); 
    data.CRttW2l_plus.passed *= (nbjet_85wp == 2 && (ht_all < 500000 || evn -> n_jets < 6)) + (nbjet_85wp >= 3 && ht_all < 500000); 
    data.CRttW2l_plus.passed *= lepton_0_ambiguity < 0 && lepton_1_ambiguity < 0; 
    data.CRttW2l_plus.passed *= (lepton_0_charge + lepton_1_charge) > 0; 
    // lepton_0_pt_GeV_NOSYS>28 
    // && (
    //      (pass_SSem_passECIDS_NOSYS && pass_SSem_passEtaCut_NOSYS) || pass_SSmm_NOSYS
    // ) 
    // && nJets>=4 
    // && ( 
    //      ( nBjets_GN2v01_85WP_NOSYS==2 && (HT_all_NOSYS < 500000 || nJets<6) ) 
    //      || 
    //      ( nBjets_GN2v01_85WP_NOSYS>=3 && HT_all_NOSYS < 500000 ) 
    // ) 
    // && lepton_0_DFCommonAddAmbiguity_NOSYS<0 
    // && lepton_1_DFCommonAddAmbiguity_NOSYS<0
    // && (lepton_0_charge_NOSYS + lepton_1_charge_NOSYS)>0 
    // name: "nJets"
    // definition: "nJets"
    // type: "int"
    // min: 3.5
    // max: 6.5
    // number_of_bins: 3


    data.CRttW2l_minus.variable1 = evn -> n_jets; 
    data.CRttW2l_minus.weight = event_weight; 
    data.CRttW2l_minus.passed *= lepton_0_pt * ((pass_ssem && pass_ssem_cut) || evn -> pass_ssmm);  
    data.CRttW2l_minus.passed *= (evn -> n_jets >= 4); 
    data.CRttW2l_minus.passed *= (nbjet_85wp == 2 && (ht_all < 500000 || evn -> n_jets < 6)) + (nbjet_85wp >= 3 && ht_all < 500000); 
    data.CRttW2l_minus.passed *= lepton_0_ambiguity < 0 && lepton_1_ambiguity < 0; 
    data.CRttW2l_minus.passed *= (lepton_0_charge + lepton_1_charge) < 0; 
    // lepton_0_pt_GeV_NOSYS>28 
    // && (
    //      (pass_SSem_passECIDS_NOSYS && pass_SSem_passEtaCut_NOSYS) || pass_SSmm_NOSYS
    // ) 
    // && nJets>=4 
    // && ( 
    //      ( nBjets_GN2v01_85WP_NOSYS == 2 && (HT_all_NOSYS < 500000 || nJets < 6) ) 
    //      || 
    //      ( nBjets_GN2v01_85WP_NOSYS >= 3 && HT_all_NOSYS<500000) 
    // ) 
    // && (lepton_0_DFCommonAddAmbiguity_NOSYS<0 && lepton_1_DFCommonAddAmbiguity_NOSYS<0) 
    // && (lepton_0_charge_NOSYS + lepton_1_charge_NOSYS) < 0
    // name: "nJets"
    // definition: "nJets"
    // type: "int"
    // min: 3.5
    // max: 6.5
    // number_of_bins: 3



    data.CR1bplus.variable1 = evn -> n_jets; 
    data.CR1bplus.weight = event_weight; 
    data.CR1bplus.passed *= lepton_0_pt; 

    blk1  = ((pass_ssee || pass_ssem) && lepton_0_ambiguity <= 0 && lepton_1_ambiguity <= 0);  
    blk1 += evn -> pass_ssmm + evn -> pass_eee_zveto + evn -> pass_eem_zveto + evn -> pass_emm_zveto + evn -> pass_mmm_zveto; 

    blk2  = (evn -> pass_eee + evn -> pass_eem + evn -> pass_emm + evn -> pass_mmm) * (sum_lepton_3_charge > 0); 
    blk2 += (evn -> pass_ssee + evn -> pass_ssmm + evn -> pass_ssem) * ((lepton_0_charge + lepton_1_charge) > 0); 
    
    data.CR1bplus.passed *= blk1 * blk2 * (ht_all > 500000.0) * (nbjet_85wp == 1) * (evn -> n_jets >= 4);

    // lepton_0_pt_GeV_NOSYS > 28 
    // && (
    //      (
    //          (pass_SSee_passECIDS_NOSYS || pass_SSem_passECIDS_NOSYS) 
    //          && lepton_0_DFCommonAddAmbiguity_NOSYS <= 0 
    //          && lepton_1_DFCommonAddAmbiguity_NOSYS <= 0
    //      )
    //      || pass_SSmm_NOSYS
    //      || pass_eee_ZVeto_NOSYS
    //      || pass_eem_ZVeto_NOSYS
    //      || pass_emm_ZVeto_NOSYS
    //      || pass_mmm_ZVeto_NOSYS
    // ) 
    // && (
    //      (
    //          (pass_eee_NOSYS || pass_eem_NOSYS || pass_emm_NOSYS || pass_mmm_NOSYS) 
    //          && (lepton_0_charge_NOSYS+lepton_1_charge_NOSYS+lepton_2_charge_NOSYS) >0 
    //      )
    //      ||
    //      (
    //          (pass_SSee_NOSYS || pass_SSmm_NOSYS || pass_SSem_NOSYS) 
    //          && (lepton_0_charge_NOSYS + lepton_1_charge_NOSYS)>0
    //      )
    // ) 
    // && HT_all_NOSYS>500000. 
    // && nBjets_GN2v01_85WP_NOSYS==1 
    // && nJets>=4
    // name: "nJets"
    // definition: "nJets"
    // type: "int"
    // min: 3.5
    // max: 10.5
    // number_of_bins: 7


    data.CR1bminus.variable1 = evn -> n_jets; 
    data.CR1bminus.weight = event_weight; 
    data.CR1bminus.passed *= lepton_0_pt; 

    blk1  = ((pass_ssee || pass_ssem) && lepton_0_ambiguity <= 0 && lepton_1_ambiguity <= 0);  
    blk1 += evn -> pass_ssmm + evn -> pass_eee_zveto + evn -> pass_eem_zveto + evn -> pass_emm_zveto + evn -> pass_mmm_zveto; 

    blk2  = (evn -> pass_eee + evn -> pass_eem + evn -> pass_emm + evn -> pass_mmm) * (sum_lepton_3_charge < 0); 
    blk2 += (evn -> pass_ssee + evn -> pass_ssmm + evn -> pass_ssem) * ((lepton_0_charge + lepton_1_charge) < 0); 
    
    data.CR1bminus.passed *= blk1 * blk2 * (ht_all > 500000.0) * (nbjet_85wp == 1) * (evn -> n_jets >= 4);
    // lepton_0_pt_GeV_NOSYS>28 
    // && (
    //      (
    //          (pass_SSee_passECIDS_NOSYS || pass_SSem_passECIDS_NOSYS) 
    //          && lepton_0_DFCommonAddAmbiguity_NOSYS<=0 
    //          && lepton_1_DFCommonAddAmbiguity_NOSYS<=0
    //      )
    //      || pass_SSmm_NOSYS
    //      || pass_eee_ZVeto_NOSYS 
    //      || pass_eem_ZVeto_NOSYS 
    //      || pass_emm_ZVeto_NOSYS 
    //      || pass_mmm_ZVeto_NOSYS
    // ) 
    // && (
    //      (
    //          (pass_eee_NOSYS || pass_eem_NOSYS || pass_emm_NOSYS || pass_mmm_NOSYS) 
    //          && (lepton_0_charge_NOSYS+lepton_1_charge_NOSYS+lepton_2_charge_NOSYS) < 0
    //      )
    //      ||
    //      (
    //          (pass_SSee_NOSYS||pass_SSmm_NOSYS||pass_SSem_NOSYS) 
    //          && (lepton_0_charge_NOSYS + lepton_1_charge_NOSYS)<0
    //      )
    // ) 
    // && HT_all_NOSYS>500000. 
    // && nBjets_GN2v01_85WP_NOSYS==1 
    // && nJets>=4"
    // name: "nJets"
    // definition: "nJets"
    // type: "int"
    // min: 3.5
    // max: 10.5
    // number_of_bins: 7

    //#######################################
    //## Validation Regions
    //#######################################

    data.CRttW2l.variable1 = lep_sum_pt; 
    data.CRttW2l.weight = event_weight;
    data.CRttW2l.passed *= (lepton_0_pt && (pass_ssem || pass_ssmm) && evn -> n_jets >= 4); 
    data.CRttW2l.passed *= ((nbjet_85wp == 2 && (evn -> HT_all < 500000.0 || evn -> n_jets < 6)) || (nbjet_85wp >= 3 && evn -> HT_all < 500000.0)); 
    data.CRttW2l.passed *= lepton_0_ambiguity < 0 && lepton_1_ambiguity < 0; 
    // lepton_0_pt_GeV_NOSYS>28 
    // && ((pass_SSem_passECIDS_NOSYS)||pass_SSmm_NOSYS) 
    // && nJets>=4 
    // && ( 
    //      (nBjets_GN2v01_85WP_NOSYS==2 && (HT_all_NOSYS<500000 || nJets<6)) 
    //      || (nBjets_GN2v01_85WP_NOSYS>=3 && HT_all_NOSYS<500000) 
    // ) 
    // && (lepton_0_DFCommonAddAmbiguity_NOSYS<0 && lepton_1_DFCommonAddAmbiguity_NOSYS<0)
    // name: "lep_pt_sum"
    // definition: "lepton_0plus1_pt_GeV_NOSYS"
    // type: "double"
    // bin_edges: [80.,100.,120.,140.,160.,180.,240.,300.]

    data.VRttZ3l.variable1 = evn -> n_jets; 
    data.VRttZ3l.variable2 = ht_all_gev; 
    data.VRttZ3l.weight = event_weight;
    data.VRttZ3l.passed = false; 
    data.VRttZ3l.passed += (evn -> pass_eee || !evn -> pass_eee_zveto); 
    data.VRttZ3l.passed += (evn -> pass_eem && !evn -> pass_eem_zveto); 
    data.VRttZ3l.passed += (evn -> pass_emm && !evn -> pass_emm_zveto); 
    data.VRttZ3l.passed += (evn -> pass_mmm && !evn -> pass_mmm_zveto); 
    data.VRttZ3l.passed *= lepton_0_pt * (nbjet_85wp >= 2) * (evn -> n_jets >= 4); 
    // lepton_0_pt_GeV_NOSYS>28 
    // && (
    //      (pass_eee_NOSYS && !pass_eee_ZVeto_NOSYS) 
    //      || (pass_eem_NOSYS && !pass_eem_ZVeto_NOSYS) 
    //      || (pass_emm_NOSYS && !pass_emm_ZVeto_NOSYS) 
    //      || (pass_mmm_NOSYS && !pass_mmm_ZVeto_NOSYS)
    // ) 
    // && nBjets_GN2v01_85WP_NOSYS>=2 
    // && nJets>=4
    // name: "nJets"
    // definition: "nJets"
    // type: "int"
    // min: 3.5
    // max: 9.5
    // number_of_bins: 6
    // name: "HT_all"
    // definition: "HT_all_GeV_NOSYS"
    // type: "double"
    // bin_edges: [500.,900.,2000.]


    data.VRttWCRSR.variable1 = evn -> n_jets; 
    data.VRttWCRSR.weight = event_weight;

    bool sub_blk1 = (pass_ssem && pass_ssem_cut) || evn -> pass_ssmm; 
    bool sub_blk2 = (nbjet_85wp == 2 && (ht_all < 500000.0 || evn -> n_jets < 6)) || (nbjet_85wp >= 3 && ht_all < 500000.0); 
    bool sub_blk3 = (evn -> pass_eee || evn -> pass_eem || evn -> pass_emm || evn -> pass_mmm) && (sum_lepton_3_charge < 0); 
    bool sub_blk4 = (evn -> pass_ssee || evn -> pass_ssmm || evn -> pass_ssem); 
    bool sub_blk5 = lepton_0_ambiguity < 0 && lepton_1_ambiguity < 0; 

    blk1  = (pass_ssee || pass_ssem || evn -> pass_ssmm || evn -> pass_eee_zveto || evn -> pass_eem_zveto || evn -> pass_emm_zveto || evn -> pass_mmm_zveto || evn -> pass_llll_zveto); 
    blk1 *= (nbjet_85wp >= 2 && evn -> n_jets >= 6 && ht_all > 500000.0); 

    data.VRttWCRSR.passed = lepton_0_pt * blk1; 
    data.VRttWCRSR.passed += (sub_blk1 && evn -> n_jets >= 4 && sub_blk2 && sub_blk5 && sub_blk3 || sub_blk4);
    data.VRttWCRSR.passed += (sub_blk1 && evn -> n_jets >= 4 && sub_blk2 && sub_blk5 && (sub_blk3 || sub_blk4));

    // lepton_0_pt_GeV_NOSYS>28 
    // && (
    //      (
    //          pass_SSee_passECIDS_NOSYS 
    //          || pass_SSem_passECIDS_NOSYS 
    //          || pass_SSmm_NOSYS 
    //          || pass_eee_ZVeto_NOSYS 
    //          || pass_eem_ZVeto_NOSYS 
    //          || pass_emm_ZVeto_NOSYS 
    //          || pass_mmm_ZVeto_NOSYS 
    //          || pass_llll_ZVeto_NOSYS
    //      ) 
    //      && nBjets_GN2v01_85WP_NOSYS >= 2 
    //      && nJets>=6 
    //      && HT_all_NOSYS>500000.
    // )
    // ||
    // (
    //     ((pass_SSem_passECIDS_NOSYS && pass_SSem_passEtaCut_NOSYS) || pass_SSmm_NOSYS) -> sub_blk1
    //     && nJets>=4 
    //     && ( (nBjets_GN2v01_85WP_NOSYS == 2 && (HT_all_NOSYS<500000 ||  nJets<6)) || (nBjets_GN2v01_85WP_NOSYS >= 3 && HT_all_NOSYS < 500000) ) -> sub_blk2
    //     && lepton_0_DFCommonAddAmbiguity_NOSYS<0 && lepton_1_DFCommonAddAmbiguity_NOSYS<0  -> sub_blk5
    //     && ( (pass_eee_NOSYS || pass_eem_NOSYS || pass_emm_NOSYS || pass_mmm_NOSYS) && (lepton_0_charge_NOSYS+lepton_1_charge_NOSYS+lepton_2_charge_NOSYS)<0  -> sub_blk3 )
    //     || (pass_SSee_NOSYS || pass_SSmm_NOSYS || pass_SSem_NOSYS) -> sub_blk4
    // )
    // ||
    // ((pass_SSem_passECIDS_NOSYS && pass_SSem_passEtaCut_NOSYS) || pass_SSmm_NOSYS) -> sub_blk1
    // && nJets>=4 
    // && ( (nBjets_GN2v01_85WP_NOSYS == 2 && (HT_all_NOSYS<500000 || nJets<6)) || (nBjets_GN2v01_85WP_NOSYS >= 3 &&  HT_all_NOSYS < 500000) ) -> sub_blk2
    // && lepton_0_DFCommonAddAmbiguity_NOSYS<0 && lepton_1_DFCommonAddAmbiguity_NOSYS<0 -> sub_blk5
    // && ( 
    //      ( (pass_eee_NOSYS || pass_eem_NOSYS || pass_emm_NOSYS || pass_mmm_NOSYS) && (lepton_0_charge_NOSYS+lepton_1_charge_NOSYS+lepton_2_charge_NOSYS)<0  -> sub_blk3 )
    //      || (pass_SSee_NOSYS || pass_SSmm_NOSYS || pass_SSem_NOSYS) -> sub_blk4
    // )
    // name: "nJets_plus_minus"
    // definition: "nJets_plus_minus_NOSYS"
    // type: "int"
    // min: 3.5
    // max: 10.5
    // number_of_bins: 7

    //#######################################
    //## Signal Regions
    //#######################################
    blk1 =  (pass_ssee || pass_ssem || evn -> pass_ssmm || evn -> pass_eee_zveto || evn -> pass_eem_zveto || evn -> pass_emm_zveto || evn -> pass_mmm_zveto || evn -> pass_llll_zveto); 
    
    data.SR4b.variable1 = ht_all_gev; 
    data.SR4b.variable2 = evn -> n_jets; 
    data.SR4b.weight = event_weight;
    data.SR4b.passed = lepton_0_pt && blk1 && nbjet_85wp >= 4 && evn -> n_jets >= 6 && ht_all > 500000.0; 
    // lepton_0_pt_GeV_NOSYS>28 
    // && (
    //      pass_SSee_passECIDS_NOSYS
    //      || pass_SSem_passECIDS_NOSYS 
    //      || pass_SSmm_NOSYS
    //      || pass_eee_ZVeto_NOSYS
    //      || pass_eem_ZVeto_NOSYS
    //      || pass_emm_ZVeto_NOSYS
    //      || pass_mmm_ZVeto_NOSYS 
    //      || pass_llll_ZVeto_NOSYS
    // ) 
    // && nBjets_GN2v01_85WP_NOSYS>=4 
    // && nJets>=6 
    // && HT_all_NOSYS>500000."
    // name: "HT_all"
    // definition: "HT_all_GeV_NOSYS"
    // type: "double"
    // bin_edges: [500.,900.,2000.]
    // name: "nJets"
    // definition: "nJets"
    // type: "int"
    // min: 5.5
    // max: 11.5
    // number_of_bins: 6

    data.SR2b.variable1 = ht_all_gev; 
    data.SR2b.variable2 = evn -> n_jets; 
    data.SR2b.weight = event_weight;
    data.SR2b.passed = lepton_0_pt && blk1 && nbjet_85wp == 2 && evn -> n_jets >= 6 && ht_all > 500000.0; 

    // lepton_0_pt_GeV_NOSYS>28 
    // && (
    //      pass_SSee_passECIDS_NOSYS
    //      || pass_SSem_passECIDS_NOSYS
    //      || pass_SSmm_NOSYS
    //      || pass_eee_ZVeto_NOSYS
    //      || pass_eem_ZVeto_NOSYS
    //      || pass_emm_ZVeto_NOSYS
    //      || pass_mmm_ZVeto_NOSYS
    //      || pass_llll_ZVeto_NOSYS
    // ) 
    // && nBjets_GN2v01_85WP_NOSYS==2 
    // && nJets>=6 
    // && HT_all_NOSYS>500000."
    // name: "HT_all"
    // definition: "HT_all_GeV_NOSYS"
    // type: "double"
    // bin_edges: [500.,700.,900.,1200.,2000.]


    data.SR3b.variable1 = ht_all_gev; 
    data.SR3b.variable2 = evn -> n_jets; 
    data.SR3b.weight = event_weight;
    data.SR3b.passed = lepton_0_pt && blk1 && nbjet_85wp == 3 && evn -> n_jets >= 6 && ht_all > 500000.0; 
    // lepton_0_pt_GeV_NOSYS > 28 
    // && (
    //      pass_SSee_passECIDS_NOSYS
    //      || pass_SSem_passECIDS_NOSYS
    //      || pass_SSmm_NOSYS
    //      || pass_eee_ZVeto_NOSYS
    //      || pass_eem_ZVeto_NOSYS
    //      || pass_emm_ZVeto_NOSYS
    //      || pass_mmm_ZVeto_NOSYS
    //      || pass_llll_ZVeto_NOSYS
    // ) 
    // && nBjets_GN2v01_85WP_NOSYS==3 
    // && nJets>=6 && HT_all_NOSYS>500000.
    // name: "HT_all"
    // definition: "HT_all_GeV_NOSYS"
    // type: "double"
    // bin_edges: [500.,700.,900.,1200.,2000.]



    data.SR2b2l.variable1 = ht_all_gev; 
    data.SR2b2l.variable2 = evn -> n_jets; 
    data.SR2b2l.passed *= (lepton_0_pt && (pass_ssee || pass_ssem || pass_ssmm)); 
    data.SR2b2l.passed *= (nbjet_85wp == 2 && evn -> n_jets >= 6 && evn -> HT_all > 500000.0); 
    data.SR2b2l.weight = event_weight;
    //  lepton_0_pt_GeV_NOSYS>28 
    //  && (
    //          pass_SSee_passECIDS_NOSYS
    //          || pass_SSem_passECIDS_NOSYS
    //          || pass_SSmm_NOSYS
    //  ) 
    //  && nBjets_GN2v01_85WP_NOSYS==2 
    //  && nJets>=6 
    //  && HT_all_NOSYS>500000.
    //  name: "HT_all"
    //  definition: "HT_all_GeV_NOSYS"
    //  type: "double"
    //  bin_edges: [500.,700.,900.,1200.,2000.]
    //  name: "nJets"
    //  definition: "nJets"
    //  type: "int"
    //  min: 5.5
    //  max: 11.5
    //  number_of_bins: 6

    data.SR2b3l4l.variable1 = ht_all_gev; 
    data.SR2b3l4l.variable2 = evn -> n_jets; 
    data.SR2b3l4l.passed *= (lepton_0_pt && (evn -> pass_eee_zveto || evn -> pass_eem_zveto || evn -> pass_emm_zveto || evn -> pass_mmm_zveto || evn -> pass_llll_zveto)); 
    data.SR2b3l4l.passed *= (nbjet_85wp == 2 && evn -> n_jets >= 6 && evn -> HT_all > 500000.0); 
    data.SR2b3l4l.weight = event_weight;
    //  lepton_0_pt_GeV_NOSYS>28 
    //  && (
    //          pass_eee_ZVeto_NOSYS
    //          || pass_eem_ZVeto_NOSYS
    //          || pass_emm_ZVeto_NOSYS
    //          || pass_mmm_ZVeto_NOSYS 
    //          || pass_llll_ZVeto_NOSYS 
    //  ) 
    //  && nBjets_GN2v01_85WP_NOSYS==2 
    //  && nJets>=6 
    //  && HT_all_NOSYS>500000.
    //  name: "HT_all"
    //  definition: "HT_all_GeV_NOSYS"
    //  type: "double"
    //  bin_edges: [500.,700.,900.,2000.]
    //  name: "nJets"
    //  definition: "nJets"
    //  type: "int"
    //  min: 5.5
    //  max: 11.5
    //  number_of_bins: 6

    data.SR2b4l.variable1 = ht_all_gev; 
    data.SR2b4l.variable2 = evn -> n_jets; 
    data.SR2b4l.passed = lepton_0_pt && evn -> pass_llll_zveto && nbjet_85wp == 2 && evn -> n_jets >= 6 && evn -> HT_all > 500000.0; 
    data.SR2b4l.weight = event_weight;
    //  lepton_0_pt_GeV_NOSYS>28 
    //  && pass_llll_ZVeto_NOSYS 
    //  && nBjets_GN2v01_85WP_NOSYS==2 
    //  && nJets>=6 
    //  && HT_all_NOSYS>500000.
    //  name: "HT_all"
    //  definition: "HT_all_GeV_NOSYS"
    //  type: "double"
    //  bin_edges: [500.,700.,900.,2000.]
    //  name: "nJets"
    //  definition: "nJets"
    //  type: "int"
    //  min: 5.5
    //  max: 11.5
    //  number_of_bins: 6


    data.SR3b2l.variable1 = ht_all_gev; 
    data.SR3b2l.variable2 = evn -> n_jets; 
    data.SR3b2l.passed *= lepton_0_pt && (pass_ssee || pass_ssem || pass_ssmm); 
    data.SR3b2l.passed *= nbjet_85wp == 3 && evn -> n_jets >= 6 && evn -> HT_all > 500000.0; 
    data.SR3b2l.weight = event_weight;
    // lepton_0_pt_GeV_NOSYS>28 
    // && (
    //      pass_SSee_passECIDS_NOSYS
    //      || pass_SSem_passECIDS_NOSYS
    //      || pass_SSmm_NOSYS
    // ) 
    // && nBjets_GN2v01_85WP_NOSYS==3 
    // && nJets>=6 
    // && HT_all_NOSYS>500000."
    // name: "HT_all"
    // definition: "HT_all_GeV_NOSYS"
    // type: "double"
    // bin_edges: [500.,700.,900.,1200.,2000.]
    // name: "nJets"
    // definition: "nJets"
    // type: "int"
    // min: 5.5
    // max: 11.5
    // number_of_bins: 6

    data.SR3b3l4l.variable1 = ht_all_gev; 
    data.SR3b3l4l.variable2 = evn -> n_jets; 
    data.SR3b3l4l.passed *= lepton_0_pt && (evn -> pass_eee_zveto || evn -> pass_eem_zveto || evn -> pass_emm_zveto || evn -> pass_mmm_zveto || evn -> pass_llll_zveto); 
    data.SR3b3l4l.passed *= nbjet_85wp == 3 && evn -> n_jets >= 6 && evn -> HT_all > 500000.0; 
    data.SR3b3l4l.weight = event_weight;
    //  lepton_0_pt_GeV_NOSYS>28 
    //  && (
    //          pass_eee_ZVeto_NOSYS
    //          || pass_eem_ZVeto_NOSYS
    //          || pass_emm_ZVeto_NOSYS
    //          || pass_mmm_ZVeto_NOSYS
    //          || pass_llll_ZVeto_NOSYS
    //  ) 
    //  && nBjets_GN2v01_85WP_NOSYS==3 
    //  && nJets>=6 
    //  && HT_all_NOSYS>500000.0
    //  name: "HT_all"
    //  definition: "HT_all_GeV_NOSYS"
    //  type: "double"
    //  bin_edges: [500.,900.,2000.]
    //  name: "nJets"
    //  definition: "nJets"
    //  type: "int"
    //  min: 5.5
    //  max: 11.5
    //  number_of_bins: 6

    data.SR3b4l.variable1 = ht_all_gev; 
    data.SR3b4l.variable2 = evn -> n_jets; 
    data.SR3b4l.passed *= lepton_0_pt && evn -> pass_llll_zveto; 
    data.SR3b4l.passed *= nbjet_85wp == 3 && evn -> n_jets >= 6 && evn -> HT_all > 500000.0; 
    data.SR3b4l.weight = event_weight;
    // lepton_0_pt_GeV_NOSYS>28 
    // && pass_llll_ZVeto_NOSYS 
    // && nBjets_GN2v01_85WP_NOSYS==3 
    // && nJets>=6 
    // && HT_all_NOSYS>500000.
    // name: "HT_all"
    // definition: "HT_all_GeV_NOSYS"
    // type: "double"
    // bin_edges: [500.,900.,2000.]
    // name: "nJets"
    // definition: "nJets"
    // type: "int"
    // min: 5.5
    // max: 11.5
    // number_of_bins: 6

    data.SR4b4l.variable1 = ht_all_gev; 
    data.SR4b4l.variable2 = evn -> n_jets; 
    data.SR4b4l.passed *= lepton_0_pt && evn -> pass_llll_zveto; 
    data.SR4b4l.passed *= nbjet_85wp >= 4 && evn -> n_jets >= 6 && evn -> HT_all > 500000.0; 
    data.SR4b4l.weight = event_weight;
    // lepton_0_pt_GeV_NOSYS>28 
    // && pass_llll_ZVeto_NOSYS 
    // && nBjets_GN2v01_85WP_NOSYS>=4 
    // && nJets>=6 
    // && HT_all_NOSYS>500000.
    // name: "HT_all"
    // definition: "HT_all_GeV_NOSYS"
    // type: "double"
    // bin_edges: [500.,620.,750.,850.,1000.,1200.,2000.]
    // name: "nJets"
    // definition: "nJets"
    // type: "int"
    // min: 5.5
    // max: 11.5
    // number_of_bins: 6

    data.SR.variable1 = ht_all_gev; 
    data.SR.variable2 = evn -> n_jets; 
    data.SR.passed *= lepton_0_pt && blk1; 
    data.SR.passed *= nbjet_85wp >= 2 && evn -> n_jets >= 6 && evn -> HT_all > 500000.0; 
    data.SR.weight = event_weight;
    // lepton_0_pt_GeV_NOSYS>28 
    // && (
    //          pass_SSee_passECIDS_NOSYS
    //          || pass_SSem_passECIDS_NOSYS
    //          || pass_SSmm_NOSYS
    //          || pass_eee_ZVeto_NOSYS
    //          || pass_eem_ZVeto_NOSYS
    //          || pass_emm_ZVeto_NOSYS
    //          || pass_mmm_ZVeto_NOSYS 
    //          || pass_llll_ZVeto_NOSYS
    // ) 
    // && nBjets_GN2v01_85WP_NOSYS>=2 
    // && nJets>=6 
    // && HT_all_NOSYS>500000.
    // name: "HT_all"
    // definition: "HT_all_GeV_NOSYS"
    // type: "double"
    // bin_edges: [500.,620.,750.,850.,1000.,1200.,2000.]
    // name: "nJets"
    // definition: "nJets"
    // type: "int"
    // min: 5.5
    // max: 11.5
    // number_of_bins: 6
    // name: "1"
    // definition: "zero_NOSYS"
    // binning:
    // min: 0
    // max: 1
    // number_of_bins: 1

    this -> output.push_back(data); 
    return true; 
}
