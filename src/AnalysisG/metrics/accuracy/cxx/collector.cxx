#include <metrics/collector.h>

pdata::pdata(){}
pdata::~pdata(){}

edata::edata(){}
edata::~edata(){
    this -> vflush(&this -> training); 
    this -> vflush(&this -> validation); 
    this -> vflush(&this -> evaluation); 
}


performance::performance(std::string mrk_, std::string mode_){
    this -> name = mrk_; this -> mode = mode_; 
}
performance::~performance(){}

rectop::rectop(){}
rectop::~rectop(){}

void rectop::decode(int chn){

}




collector::collector(){}
collector::~collector(){}

std::string collector::label(std::vector<pdata*>* vl, std::string mox){
    for (size_t x(0); x < vl -> size(); ++x){
        if (!vl -> at(x)){return "";}
        return mox;
    } 
    return ""; 
}

void collector::inlet(edata* data){
    auto expand = [](pdata* px){
        for (size_t x(0); x < px -> pt.size(); ++x){
            rectop* rtx = new rectop(); 
            rtx -> pt  = px -> pt[x]; 
            rtx -> eta = px -> eta[x]; 
            rtx -> phi = px -> phi[x];
            if (px -> energy.size()){rtx -> e = px -> energy[x];} 
            if (px -> mass.size()){rtx -> mass = px -> mass[x];}
        }
    }; 


    std::string mode = ""; 
    mode += this -> label(&data -> training  , "training"  ); 
    mode += this -> label(&data -> validation, "validation"); 
    mode += this -> label(&data -> evaluation, "evaluation"); 
  
    std::map<std::string, performance*>* prf = nullptr;  
    if (mode == "training"  ){prf = &this -> training;  }
    if (mode == "validation"){prf = &this -> validation;}
    if (mode == "evaluation"){prf = &this -> evaluation;}
    if (!prf -> count(data -> mrk)){(*prf)[data -> mrk] = new performance(data -> mrk, mode);}

    performance* pr = (*prf)[data -> mrk];
    edata_t ev = edata_t();
    ev.ntru = data -> ntop_tru;
    ev.npre = data -> ntop_prd;
    ev.pred = data -> ntop_score; 
    ev.acc_edges = data -> acc_edge; 

//    particle_tps pnx = particle_tps();
//    for (size_t x(0); x < data -> training.size(); ++x){
//        expand(&data -> training[x]); 
//    }





//    pr -> epoch[data -> epoch][data -> kfold].push_back(




}


