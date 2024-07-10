#include "childrenkinematics.h"

childrenkinematics::childrenkinematics(){this -> name = "childrenkinematics";}
childrenkinematics::~childrenkinematics(){}

selection_template* childrenkinematics::clone(){
    return (selection_template*)new childrenkinematics();
}

void childrenkinematics::merge(selection_template* sl){
    childrenkinematics* slt = (childrenkinematics*)sl; 

    merge_data(&this -> res_kinematics       , &slt -> res_kinematics);
    merge_data(&this -> spec_kinematics      , &slt -> spec_kinematics); 
    merge_data(&this -> res_pdgid_kinematics , &slt -> res_pdgid_kinematics);
    merge_data(&this -> spec_pdgid_kinematics, &slt -> spec_pdgid_kinematics);
    merge_data(&this -> res_decay_mode       , &slt -> res_decay_mode);
    merge_data(&this -> spec_decay_mode      , &slt -> spec_decay_mode);
    merge_data(&this -> mass_clustering      , &slt -> mass_clustering);
    merge_data(&this -> fractional           , &slt -> fractional);
    merge_data(&this -> dr_clustering        , &slt -> dr_clustering);
    merge_data(&this -> top_pt_clustering    , &slt -> top_pt_clustering);
    merge_data(&this -> top_energy_clustering, &slt -> top_energy_clustering);
    merge_data(&this -> top_children_dr      , &slt -> top_children_dr);
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
        particle_template* ch_ = res_children[x]; 
        this -> dump_kinematics(&this -> res_kinematics, ch_); 
        this -> dump_kinematics(&this -> res_pdgid_kinematics[ch_ -> symbol], ch_); 
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
            this -> dump_kinematics(&this -> res_decay_mode[flag], c_); 
            std::string fg = "r" + flag;
            this -> top_children_dr[fg].push_back(t -> DeltaR(c_));
            this -> fractional[fg + "-pt"][c_ -> symbol].push_back(c_ -> pt / t -> pt); 
            this -> fractional[fg + "-energy"][c_ -> symbol].push_back(c_ -> e / t -> e); 
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
        particle_template* ch_ = spec_children[x]; 
        this -> dump_kinematics(&this -> spec_kinematics, ch_); 
        this -> dump_kinematics(&this -> spec_pdgid_kinematics[ch_ -> symbol], ch_); 
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
            this -> dump_kinematics(&this -> spec_decay_mode[flag], c_); 
            std::string fg = "s" + flag;
            this -> top_children_dr[fg].push_back(t -> DeltaR(c_));
            this -> fractional[fg + "-pt"][c_ -> symbol].push_back(c_ -> pt / t -> pt); 
            this -> fractional[fg + "-energy"][c_ -> symbol].push_back(c_ -> e / t -> e); 
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

            std::string flag = ""; 
            if ( t1_ -> from_res &&  t2_ -> from_res){flag = "RR";}
            else if ( t1_ -> from_res && !t2_ -> from_res){flag = "RS";}
            else if (!t1_ -> from_res &&  t2_ -> from_res){flag = "RS";}
            else if (!t1_ -> from_res && !t2_ -> from_res){flag = "SS";}
            
            if ((*t1_) == (*t2_)){flag = "CT" + flag;}
            else {flag = "FT" + flag;}

            this -> dr_clustering[flag].push_back(c1_ -> DeltaR(c2_)); 
            this -> top_pt_clustering[flag].push_back(t2_ -> pt / 1000); 
            this -> top_energy_clustering[flag].push_back(t2_ -> e / 1000);

            std::map<std::string, particle_template*> ch_1 = t1_ -> children; 
            std::map<std::string, particle_template*> ch_2 = t2_ -> children; 
            std::vector<particle_template*> c_1 = this -> vectorize(&ch_1); 
            std::vector<particle_template*> c_2 = this -> vectorize(&ch_2); 

            std::vector<particle_template*> pair_mass = {}; 
            merge_data(&pair_mass, &c_1); 
            merge_data(&pair_mass, &c_2); 
            this -> mass_clustering[flag].push_back(this -> sum(&pair_mass)); 
        }
    }
    return true; 
}

