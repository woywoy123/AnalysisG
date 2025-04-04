#include "childrenkinematics.h"

childrenkinematics::childrenkinematics(){this -> name = "childrenkinematics";}
childrenkinematics::~childrenkinematics(){}

selection_template* childrenkinematics::clone(){
    return (selection_template*)new childrenkinematics();
}

void childrenkinematics::merge(selection_template* sl){
    childrenkinematics* slt = (childrenkinematics*)sl; 
    
    auto lambK = [this](std::vector<kinematic_t>* ptr, std::string key){
        std::vector<float> pt, e, eta, phi;
        std::vector<int> pdgid;
        for (size_t x(0); x < ptr -> size(); ++x){
            pt.push_back((*ptr)[x].pt); 
             e.push_back((*ptr)[x].energy); 
            eta.push_back((*ptr)[x].eta); 
            phi.push_back((*ptr)[x].phi); 
            pdgid.push_back((*ptr)[x].pdgid); 
        }
        this -> write(&pt   , key + "_pt");
        this -> write(&e    , key + "_energy"); 
        this -> write(&eta  , key + "_eta");
        this -> write(&phi  , key + "_phi"); 
        this -> write(&pdgid, key + "_pdgid");
    }; 

    auto lambM = [this](std::vector<misc_t>* ptr, std::string key){
        std::vector<int> pdgid;
        std::vector<bool> is_lep, is_res; 
        std::vector<float> pt, e, eta, phi, mass, dR, frc_e, frc_pt;
        for (size_t x(0); x < ptr -> size(); ++x){
            pt.push_back((*ptr)[x].kin.pt); 
             e.push_back((*ptr)[x].kin.energy); 
            eta.push_back((*ptr)[x].kin.eta); 
            phi.push_back((*ptr)[x].kin.phi); 
            pdgid.push_back((*ptr)[x].kin.pdgid); 

            is_lep.push_back((*ptr)[x].is_lep); 
            is_res.push_back((*ptr)[x].is_res); 
            mass.push_back((*ptr)[x].mass_clust); 
            dR.push_back((*ptr)[x].delta_R); 
            frc_e.push_back((*ptr)[x].frc_energy); 
            frc_pt.push_back((*ptr)[x].frc_pt); 
        }
        
        this -> write(&pt    , key + "_decay_pt");
        this -> write(&e     , key + "_decay_energy");
        this -> write(&eta   , key + "_decay_eta");
        this -> write(&phi   , key + "_decay_phi"); 
        this -> write(&pdgid , key + "_decay_pdgid");
        this -> write(&is_lep, key + "_decay_islep");
        this -> write(&is_res, key + "_decay_isres");
        this -> write(&mass  , key + "_decay_mass");
        this -> write(&dR    , key + "_decay_dR");
        this -> write(&frc_e , key + "_decay_frc_e");
        this -> write(&frc_pt, key + "_decay_frc_pt");
    }; 

    auto lambP = [this](std::vector<perms_t>* ptr, std::string key){
        std::vector<bool> rr, ss, rs, ct, ft; 
        std::vector<float> pt, dR, e, mass;
        for (size_t x(0); x < ptr -> size(); ++x){
            pt.push_back((*ptr)[x].top_pt); 
             e.push_back((*ptr)[x].top_e); 
            dR.push_back((*ptr)[x].delta_R); 
            mass.push_back((*ptr)[x].mass); 
            rr.push_back((*ptr)[x].RR); 
            ss.push_back((*ptr)[x].SS); 
            rs.push_back((*ptr)[x].RS); 
            ct.push_back((*ptr)[x].CT); 
            ft.push_back((*ptr)[x].FT); 
        }
        this -> write(&pt  , key + "_pt");
        this -> write(&e   , key + "_energy"); 
        this -> write(&mass, key + "_mass");
        this -> write(&dR  , key + "_dR");
        this -> write(&rr  , key + "_RR");
        this -> write(&ss  , key + "_SS");
        this -> write(&rs  , key + "_RS");
        this -> write(&ct  , key + "_CT");
        this -> write(&ft  , key + "_FT");
    };  

    lambK(&slt -> res_kinematics, "res");
    lambK(&slt -> spec_kinematics, "spec");
    lambM(&slt -> res_decay_mode, "res");
    lambM(&slt -> spec_decay_mode, "spec");
    lambP(&slt -> top_clusters, "top_perm");


    //merge_data(&this -> res_kinematics       , &slt -> res_kinematics);
    //merge_data(&this -> spec_kinematics      , &slt -> spec_kinematics); 
    //merge_data(&this -> res_pdgid_kinematics , &slt -> res_pdgid_kinematics);
    //merge_data(&this -> spec_pdgid_kinematics, &slt -> spec_pdgid_kinematics);
    //merge_data(&this -> res_decay_mode       , &slt -> res_decay_mode);
    //merge_data(&this -> spec_decay_mode      , &slt -> spec_decay_mode);
    //merge_data(&this -> mass_clustering      , &slt -> mass_clustering);
    //merge_data(&this -> fractional           , &slt -> fractional);
    //merge_data(&this -> dr_clustering        , &slt -> dr_clustering);
    //merge_data(&this -> top_pt_clustering    , &slt -> top_pt_clustering);
    //merge_data(&this -> top_energy_clustering, &slt -> top_energy_clustering);
    //merge_data(&this -> top_children_dr      , &slt -> top_children_dr);
}


bool childrenkinematics::selection(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
    std::vector<particle_template*> tops = evn -> Tops; 
    int n_res = 0; 
    int n_spec = 0; 

    for (size_t x(0); x < tops.size(); ++x){
        top* t = (top*)tops[x]; 
        if (t -> from_res){n_res++;}
        else {n_spec++;}
    }
    return (n_res == 2) && (n_spec == 2);
}

bool childrenkinematics::strategy(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
    std::vector<particle_template*> tops = evn -> Tops;  

    std::vector<particle_template*> res_t = {}; 
    std::vector<particle_template*> res_children = {}; 
    for (size_t x(0); x < tops.size(); ++x){
        top* t = (top*)tops[x]; 
        if (!t -> from_res){continue;}
        res_t.push_back(tops[x]); 
        std::map<std::string, particle_template*> ch = t -> children;
        std::vector<particle_template*> ch_ = this -> vectorize(&ch); 
        merge_data(&res_children, &ch_); 
    }
    for (size_t x(0); x < res_children.size(); ++x){
        this -> dump_kinematics(&this -> res_kinematics, res_children[x]); 
    }

    for (size_t x(0); x < res_t.size(); ++x){
        top* t = (top*)res_t[x]; 

        std::string flag = "had"; 
        std::map<std::string, particle_template*> ch = t -> children; 
        std::vector<particle_template*> ch_ = this -> vectorize(&ch); 
        for (size_t c(0); c < ch_.size(); ++c){
            if (!ch_[c] -> is_lep){continue;}
            flag = "lep"; 
            break; 
        }        

        for (size_t c(0); c < ch_.size(); ++c){
            particle_template* c_ = ch_[c];

            misc_t msx; 
            msx.is_res = true;
            msx.is_lep = (flag == "lep"); 
            this -> dump_kinematics(&msx.kin, c_); 
            msx.delta_R = t -> DeltaR(c_); 
            msx.frc_pt = (c_ -> pt / t -> pt); 
            msx.frc_energy = (c_ -> e / t -> e); 
            this -> res_decay_mode.push_back(msx); 
        }
    }

    std::vector<particle_template*> spec_t = {}; 
    std::vector<particle_template*> spec_children = {}; 
    for (size_t x(0); x < tops.size(); ++x){
        top* t = (top*)tops[x]; 
        if (t -> from_res){continue;}
        spec_t.push_back(tops[x]); 
        std::map<std::string, particle_template*> ch = t -> children;
        std::vector<particle_template*> ch_ = this -> vectorize(&ch); 
        merge_data(&spec_children, &ch_); 
    }

    for (size_t x(0); x < spec_children.size(); ++x){
        this -> dump_kinematics(&this -> spec_kinematics, spec_children[x]); 
    }

    for (size_t x(0); x < spec_t.size(); ++x){
        top* t = (top*)spec_t[x]; 

        std::string flag = "had"; 
        std::map<std::string, particle_template*> ch = t -> children; 
        std::vector<particle_template*> ch_ = this -> vectorize(&ch); 
        for (size_t c(0); c < ch_.size(); ++c){
            if (!ch_[c] -> is_lep){continue;}
            flag = "lep"; 
            break; 
        }        

        for (size_t c(0); c < ch_.size(); ++c){
            particle_template* c_ = ch_[c];

            misc_t msx; 
            msx.is_res = false;
            msx.is_lep = (flag == "lep"); 
            this -> dump_kinematics(&msx.kin, c_); 
            msx.delta_R = t -> DeltaR(c_); 
            msx.frc_pt = (c_ -> pt / t -> pt); 
            msx.frc_energy = (c_ -> e / t -> e); 
            this -> spec_decay_mode.push_back(msx); 
        }
    }

    std::vector<std::vector<particle_template*>> pairs = {}; 
    for (size_t x(0); x < tops.size(); ++x){
        std::map<std::string, particle_template*> ch_ = tops[x] -> children;
        std::vector<particle_template*> ch = this -> vectorize(&ch_); 
        for (size_t c(0); c < ch.size(); ++c){pairs.push_back({tops[x], ch[c]});}
    }

    for (size_t t1(0); t1 < pairs.size(); ++t1){
        for (size_t t2(t1+1); t2 < pairs.size(); ++t2){
            top* t1_ = (top*)pairs[t1][0];
            top* t2_ = (top*)pairs[t2][0]; 

            particle_template* c1_ = pairs[t1][1]; 
            particle_template* c2_ = pairs[t2][1];  

            perms_t xt; 
            xt.RR  = ( t1_ -> from_res) * ( t2_ -> from_res); 
            xt.RS  = ( t1_ -> from_res) * (!t2_ -> from_res); 
            xt.RS += (!t1_ -> from_res) * ( t2_ -> from_res); 
            xt.SS  = (!t1_ -> from_res) * (!t2_ -> from_res); 
            xt.CT  =  ((*t1_) == (*t2_)); 
            xt.FT  = !((*t1_) == (*t2_)); 
            xt.delta_R = c1_ -> DeltaR(c2_); 

            xt.top_pt = t2_ -> pt / 1000; 
            xt.top_e  = t2_ -> e  / 1000; 

            std::map<std::string, particle_template*> ch_1 = t1_ -> children; 
            std::map<std::string, particle_template*> ch_2 = t2_ -> children; 
            std::vector<particle_template*> c_1 = this -> vectorize(&ch_1); 
            std::vector<particle_template*> c_2 = this -> vectorize(&ch_2); 
            std::vector<particle_template*> pair_mass = {}; 
            merge_data(&pair_mass, &c_1); 
            merge_data(&pair_mass, &c_2); 
            xt.mass  = this -> sum(&pair_mass); 
            this -> top_clusters.push_back(xt);
        }
    }
    return true; 
}

