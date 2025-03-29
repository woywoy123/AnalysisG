#include "matching.h"

matching::matching(){this -> name = "matching";}
selection_template* matching::clone(){
    matching* ptr = new matching(); 
    ptr -> energy_constraint = this -> energy_constraint;     
    return (selection_template*)ptr; 
}
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

    this -> write(&slt -> data.top_truthjets.is_leptonic  , "tj_isleptonic"); 
    this -> write(&slt -> data.top_truthjets.is_hadronic  , "tj_ishadronic"); 
    this -> write(&slt -> data.top_truthjets.pdgid        , "tj_pdgid"); 
    this -> write(&slt -> data.top_truthjets.num_jets     , "tj_num_jets"); 
    this -> write(&slt -> data.top_truthjets.merged       , "tj_merged_top_jets"); 
    
    this -> write(&slt -> data.top_truthjets.wrong_matched, "tj_wrong_match"); 
    this -> write(&slt -> data.top_truthjets.num_false    , "tj_num_false"); 

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
    
    this -> write(&slt -> data.top_jets_children.wrong_matched, "jc_wrong_match"); 
    this -> write(&slt -> data.top_jets_children.num_false    , "jc_num_false"); 

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

    this -> write(&slt -> data.top_jets_leptons.wrong_matched, "jl_wrong_match"); 
    this -> write(&slt -> data.top_jets_leptons.num_false    , "jl_num_false"); 
}

void matching::dump(
        object_data_t* datav, std::vector<particle_template*>* obj, 
        bool is_lepx, bool is_lep_tru, int* num_jets, std::vector<int>* num_merged
){
    if (!obj -> size()){return;}
    bool wrg = is_lepx != is_lep_tru; 

    datav -> num_tops += 1; 
    datav -> num_false += wrg; 
    datav -> num_ltop += (!wrg)*( is_lepx); 
    datav -> num_htop += (!wrg)*(!is_lepx); 

    datav -> is_leptonic.push_back(int( is_lepx)*(!wrg) - 1*(wrg)); 
    datav -> is_hadronic.push_back(int(!is_lepx)*(!wrg) - 1*(wrg)); 

    if (num_jets){datav -> num_jets.push_back(*num_jets);}
    if (num_merged){datav -> merged.push_back(*num_merged);}
    datav -> pdgid.push_back(this -> get_pdgid(obj)); 
  
    double mx = this -> sum(obj); 
    if (is_lepx != is_lep_tru){datav -> wrong_matched.push_back(mx);}
    else {datav -> mass.push_back(mx);}
}

bool matching::match_obj(
        std::vector<particle_template*>* vx, std::vector<particle_template*>* out, 
        std::string hash_, std::vector<int>* num_merged, int* num_jets, bool exl_lep
){

    auto lmb = [](std::map<std::string, double>* in){
            std::vector<double> out = {}; 
            std::map<std::string, double>::iterator itr = in -> begin(); 
            for (; itr != in -> end(); ++itr){out.push_back(itr -> second);}
            return out; 
    }; 

    *num_jets = 0; 
    out -> clear(); 
    num_merged -> clear(); 
    bool is_lepx = false; 
    std::map<std::string, double> frac = {}; 
    std::map<std::string, particle_template*> tmp = {}; 

    for (size_t y(0); y < vx -> size(); ++y){
        particle_template* ptr = (*vx)[y]; 
        std::map<std::string, particle_template*> prnt = ptr -> parents; 
        if (!prnt.count(hash_)){continue;}
        if (exl_lep && ptr -> is_lep){is_lepx = true; continue;}
        if (ptr -> is_lep){is_lepx = true;}
        std::string hashx = ptr -> hash; 
        frac[hashx] = ptr -> e / 1000.0; 
        tmp[hashx] = ptr;
    }
    std::vector<double> energy_frac = lmb(&frac); 
    std::vector<particle_template*> tmp_v = this -> vectorize(&tmp);

    double tl = this -> tools::sum(&energy_frac);
    for (size_t y(0); y < energy_frac.size(); ++y){
        particle_template* tx = tmp_v[y]; 
        std::map<std::string, particle_template*> prnt = tx -> parents; 
        if (energy_frac[y] / tl < this -> energy_constraint){continue;}
        out -> push_back(tx); 
        if (tx -> is_lep){continue;}
        num_merged -> push_back(int(prnt.size()));  
        *num_jets += 1; 
    }
    return is_lepx; 
}


std::vector<int> matching::get_pdgid(std::vector<particle_template*>* prt){
    std::vector<int> out; 
    for (size_t x(0); x < prt -> size(); ++x){out.push_back((*prt)[x] -> pdgid);}
    return out; 
}


bool matching::strategy(event_template* ev){
    std::string evnt = ev -> name; 
    if (evnt == "experimental_mc20"){this -> experimental(ev);}
    if (evnt == "ssml_mc20"){this -> current(ev);}
    if (evnt == "bsm_4tops"){this -> reference(ev);}
    return true; 
}
