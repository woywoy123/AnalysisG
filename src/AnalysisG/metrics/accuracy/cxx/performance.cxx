#include <metrics/collector.h>
#include <metrics/transfer.h>

performance::performance(std::string mrk_, std::string mode_){
    this -> name = mrk_; 
    this -> mode = mode_; 
}

void performance::compiler(){
    std::map<int, std::map<int, std::vector<edata_t>>>::iterator itx;
    for (itx = this -> epoch.begin(); itx != this -> epoch.end(); ++itx){
        std::map<int, std::vector<edata_t>>::iterator ity = itx -> second.begin();
        for (; ity != itx -> second.end(); ++ity){
            if (!ity -> second.size()){continue;}
            this -> metric[itx -> first][ity -> first] = this -> filter(&ity -> second);
            ity -> second.clear(); 
            ity -> second.shrink_to_fit(); 
        }
    }
}

performance_t* performance::filter(std::vector<edata_t>* dfc){
    performance_t* pf = new performance_t(); 
    
    for (size_t x(0); x < dfc -> size(); ++x){
        edata_t* dt = &dfc -> at(x); 
        std::string prc = process_string(dt -> prc); 

        // =====================[ TRUTH ]=================== //
        int ntr = 0; 
        pairs_t* pdx = &pf -> truth[prc]; 
        for (size_t y(0); y < dt -> truth.size(); ++y){
            rectop* dp = &dt -> truth[y]; 
            pdx -> raw.mass(dp);

            if (!dp -> valid()){continue;}
            pdx -> adj.mass(dp); 
            ntr++; 
        }

        pdx -> raw.mlps(&dt -> pred); 
        pdx -> raw.assign(dt -> ntru, dt -> npre); 
        pdx -> adj.assign(ntr       , dt -> npre); 
        // ================================================= //

        // ==================[ NOMINAL ]==================== //
        int npa = 0; 
        int npr = dt -> nominal.size();
        pdx = &pf -> nominal[prc]; 
        for (size_t y(0); y < dt -> nominal.size(); ++y){
            rectop* dp = &dt -> nominal[y]; 
            pdx -> raw.mass(dp);
            pdx -> raw.weights(dp -> score);

            if (!dp -> valid()){continue;}
            pdx -> adj.mass(dp); 
            pdx -> adj.weights(dp -> score);
            npa++; 
        }
        pdx -> raw.assign(dt -> ntru, npr);  
        pdx -> adj.assign(ntr       , npa); 
        // ================================================= //

        // ===================[ UnMasked ]================== //
        npa = 0; 
        npr = dt -> umasked.size();
        pdx = &pf -> unmasked[prc]; 
        for (size_t y(0); y < dt -> umasked.size(); ++y){
            rectop* dp = &dt -> umasked[y]; 
            pdx -> raw.mass(dp);
            pdx -> raw.weights(dp -> score);

            if (!dp -> valid()){continue;}
            pdx -> adj.mass(dp); 
            pdx -> adj.weights(dp -> score);
            npa++; 
        }
        pdx -> raw.assign(dt -> ntru, npr);  
        pdx -> adj.assign(ntr       , npa); 
        // ================================================ //

        // ===================[ Masked ]=================== //
        npa = 0; 
        npr = dt -> masked.size();
        pdx = &pf -> masked[prc]; 
        for (size_t y(0); y < dt -> masked.size(); ++y){
            rectop* dp = &dt -> masked[y]; 
            pdx -> raw.mass(dp);
            pdx -> raw.weights(dp -> score);

            if (!dp -> valid()){continue;}
            pdx -> adj.mass(dp); 
            pdx -> adj.weights(dp -> score);
            npa++; 
        }
        pdx -> raw.assign(dt -> ntru, npr);  
        pdx -> adj.assign(ntr       , npa); 
        // ================================================ //
    }
    return pf; 
}

performance::~performance(){
    std::map<int, std::map<int, performance_t*>>::iterator itx = this -> metric.begin();
    for (; itx != this -> metric.end(); ++itx){
        std::map<int, performance_t*>::iterator ity = itx -> second.begin();
        for (; ity != itx -> second.end(); ++ity){
            if (ity -> second){
                delete ity -> second;
                ity -> second = nullptr;
            }
        }
    }
}


edata::edata(){}
edata::~edata(){
    this -> vflush(&this -> training); 
    this -> vflush(&this -> validation); 
    this -> vflush(&this -> evaluation); 
}

evn_t::evn_t(){
    for (size_t x(0); x < 25; ++x){
        this -> matrix[x / 5][x % 5] = 0;
    }
}

void evn_t::assign(int tru, int pred){
    this -> matrix[tru][pred]++; 
    this -> ntops.push_back(pred);
    this -> ntrus.push_back(tru); 
}

void evn_t::mass(rectop* vl){
    this -> masses.push_back(vl -> mass);
    this -> index.push_back(vl -> index);

}

void evn_t::weights(double w){this -> weight.push_back(w);}

void evn_t::weights(std::vector<double>* vec){
    for (int x(0); x < vec -> size(); ++x){
        this -> weights(vec -> at(x));
    }
}

void evn_t::mlps(std::vector<double>* vec){
    this -> mlp.push_back(*vec); 
}

