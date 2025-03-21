#include "matching.h"

matching::matching(){this -> name = "matching";}
selection_template* matching::clone(){return (selection_template*)new matching();}
matching::~matching(){}

void matching::merge(selection_template* sl){
    matching* slt = (matching*)sl; 

    // ------------------ top-partons ------------------  
    this -> write(&slt -> data.top_partons.num_tops, "p_ntops"); 
    this -> write(&slt -> data.top_partons.num_ltop, "p_ltops");
    this -> write(&slt -> data.top_partons.num_htop, "p_htops");
    this -> write(&slt -> data.top_partons.mass    , "p_top_mass"); 

    // --------------- top-children ------------------
    this -> write(&slt -> data.top_children.num_tops, "c_ntops"); 
    this -> write(&slt -> data.top_children.num_ltop, "c_ltops");
    this -> write(&slt -> data.top_children.num_htop, "c_htops");
    this -> write(&slt -> data.top_children.mass    , "c_top_mass"); 

    this -> write(&slt -> data.top_children.is_leptonic, "c_isleptonic"); 
    this -> write(&slt -> data.top_children.is_hadronic, "c_ishadronic"); 
    this -> write(&slt -> data.top_children.pdgid      , "c_pdgid"); 


    // ---------------------- truth jets ------------------------- 
    this -> write(&slt -> data.top_truthjets.num_tops, "tj_ntops"); 
    this -> write(&slt -> data.top_truthjets.num_ltop, "tj_ltops");
    this -> write(&slt -> data.top_truthjets.num_htop, "tj_htops");
    this -> write(&slt -> data.top_truthjets.mass    , "tj_top_mass"); 

    this -> write(&slt -> data.top_truthjets.is_leptonic, "tj_isleptonic"); 
    this -> write(&slt -> data.top_truthjets.is_hadronic, "tj_ishadronic"); 
    this -> write(&slt -> data.top_truthjets.pdgid      , "tj_pdgid"); 
    this -> write(&slt -> data.top_truthjets.num_jets   , "tj_num_jets"); 
    this -> write(&slt -> data.top_truthjets.merged     , "tj_merged_top_jets"); 



    // ------------- jets with top children leptons and neutrinos -----------
    this -> write(&slt -> data.top_jets_children.num_tops, "jc_ntops"); 
    this -> write(&slt -> data.top_jets_children.num_ltop, "jc_ltops");
    this -> write(&slt -> data.top_jets_children.num_htop, "jc_htops");
    this -> write(&slt -> data.top_jets_children.mass    , "jc_top_mass"); 

    this -> write(&slt -> data.top_jets_children.is_leptonic, "jc_isleptonic"); 
    this -> write(&slt -> data.top_jets_children.is_hadronic, "jc_ishadronic"); 
    this -> write(&slt -> data.top_jets_children.pdgid      , "jc_pdgid"); 
    this -> write(&slt -> data.top_jets_children.num_jets   , "jc_num_jets"); 
    this -> write(&slt -> data.top_jets_children.merged     , "jc_merged_top_jets"); 


    // ----------- Detector jets with leptons and top children neutrinos ----------------
    this -> write(&slt -> data.top_jets_leptons.num_tops, "jl_ntops"); 
    this -> write(&slt -> data.top_jets_leptons.num_ltop, "jl_ltops");
    this -> write(&slt -> data.top_jets_leptons.num_htop, "jl_htops");
    this -> write(&slt -> data.top_jets_leptons.mass    , "jl_top_mass"); 

    this -> write(&slt -> data.top_jets_leptons.is_leptonic, "jl_isleptonic"); 
    this -> write(&slt -> data.top_jets_leptons.is_hadronic, "jl_ishadronic"); 
    this -> write(&slt -> data.top_jets_leptons.pdgid      , "jl_pdgid"); 
    this -> write(&slt -> data.top_jets_leptons.num_jets   , "jl_num_jets"); 
    this -> write(&slt -> data.top_jets_leptons.merged     , "jl_merged_top_jets"); 
}

bool matching::strategy(event_template* ev){
    std::string evnt = ev -> name; 
    if (evnt == "experimental_mc20"){this -> experimental(ev);}
//    if (evnt == "ssml_mc20"){this -> current(ev);}
//    if (evnt == "bsm_4tops"){this -> reference(ev);}
    return true; 
}
