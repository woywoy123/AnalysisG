#include "topkinematics.h"

topkinematics::topkinematics(){this -> name = "top-kinematics";}
topkinematics::~topkinematics(){}

selection_template* topkinematics::clone(){
    return (selection_template*)new topkinematics();
}

void topkinematics::merge(selection_template* sl){
    topkinematics* slt = (topkinematics*)sl; 

    merge_data(&this -> res_top_kinematics , &slt -> res_top_kinematics); 
    merge_data(&this -> spec_top_kinematics, &slt -> spec_top_kinematics); 
    merge_data(&this -> mass_combi         , &slt -> mass_combi); 
    merge_data(&this -> deltaR             , &slt -> deltaR); 
}

bool topkinematics::selection(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
    std::vector<particle_template*>* tops = &evn -> Tops; 
    int res = 0; 
    for (size_t x(0); x < tops -> size(); ++x){res += ((top*)tops -> at(x)) -> from_res;} 
    return res == 2; 
}

bool topkinematics::strategy(event_template* ev){
    auto kins = [](std::map<std::string, std::vector<float>>* data, std::vector<top*>* td){
        for (size_t x(0); x < td -> size(); ++x){
            (*data)["pt"].push_back(td -> at(x) -> pt / 1000); 
            (*data)["eta"].push_back(td -> at(x) -> eta); 
            (*data)["phi"].push_back(td -> at(x) -> phi); 
            (*data)["energy"].push_back(td -> at(x) -> e / 1000); 
        }
    };

    bsm_4tops* evn = (bsm_4tops*)ev; 
    std::vector<top*> r_tops = {};
    std::vector<top*> s_tops = {}; 
    for (size_t x(0); x < evn -> Tops.size(); ++x){
        top* t = (top*)evn -> Tops[x]; 
        if (t -> from_res){r_tops.push_back(t);}
        else {s_tops.push_back(t);}
    }

    kins(&this -> res_top_kinematics , &r_tops); 
    kins(&this -> spec_top_kinematics, &s_tops); 

    std::vector<float>* data = nullptr; 

    data = &this -> deltaR["RS"]; 
    for (size_t x(0); x < r_tops.size(); ++x){
        for (size_t y(0); y < s_tops.size(); ++y){
            data -> push_back(r_tops[x] -> DeltaR(s_tops[y])); 
        }
    }

    data = &this -> deltaR["RR"]; 
    for (size_t x(0); x < r_tops.size(); ++x){
        for (size_t y(x); y < r_tops.size(); ++y){
            if (x == y){continue;}
            data -> push_back(r_tops[x] -> DeltaR(r_tops[y])); 
        }
    }

    data = &this -> deltaR["SS"]; 
    for (size_t x(0); x < s_tops.size(); ++x){
        for (size_t y(x); y < s_tops.size(); ++y){
            if (x == y){continue;}
            data -> push_back(s_tops[x] -> DeltaR(s_tops[y])); 
        }
    }
 
    data = &this -> mass_combi["RS"]; 
    for (size_t x(0); x < r_tops.size(); ++x){
        for (size_t y(0); y < s_tops.size(); ++y){
            top r = (*(r_tops[x]) + *s_tops[y]); 
            data -> push_back(r.mass / 1000); 
        }
    }

    data = &this -> mass_combi["RR"]; 
    for (size_t x(0); x < r_tops.size(); ++x){
        for (size_t y(x); y < r_tops.size(); ++y){
            if (x == y){continue;}
            top r = (*(r_tops[x]) + *r_tops[y]); 
            data -> push_back(r.mass / 1000); 
        }
    }

    data = &this -> mass_combi["SS"]; 
    for (size_t x(0); x < s_tops.size(); ++x){
        for (size_t y(x); y < s_tops.size(); ++y){
            if (x == y){continue;}
            top r = (*(s_tops[x]) + *s_tops[y]); 
            data -> push_back(r.mass / 1000); 
         }
    }

    return true; 
}
