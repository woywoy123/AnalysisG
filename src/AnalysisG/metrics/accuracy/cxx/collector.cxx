#include <metrics/collector.h>

pdata::pdata(){}
pdata::~pdata(){}

rectop::rectop(){}
rectop::~rectop(){}

void rectop::decode(int _chn){
    int xhn = std::abs(_chn); 
    int n = xhn / 1000;
    int l = (xhn - n * 1000) / 100;
    int b = (xhn - n * 1000 - l * 100) / 10; 
    int p = (xhn - n * 1000 - l * 100 - b * 10); 

    this -> composition[content_e::other]  = n;
    this -> composition[content_e::lepton] = l;
    this -> composition[content_e::bquark] = b;

    if (l >= 1 && l < 3 && b == 1 && p == 2){p = 2;} 
    else {p = 0;}

    this -> idn = (p == 2) ? object_e::valid : object_e::failed;
    if (_chn < 0 && int(p / 2)){this -> chn = channel_e::hadronic;}
    if (_chn > 0 && int(p / 2)){this -> chn = channel_e::leptonic;}
}

bool rectop::valid(){return this -> idn == object_e::valid;}


collector::collector(){}
collector::~collector(){}
void collector::flush(){
    this -> mflush(&this -> training); 
    this -> mflush(&this -> validation);
    this -> mflush(&this -> evaluation); 
}

std::string collector::label(std::vector<pdata*>* vl, std::string mox){
    for (size_t x(0); x < vl -> size(); ++x){
        if (!vl -> at(x)){return "";}
        return mox;
    } 
    return ""; 
}


void collector::expand(edata_t* ev, pdata* px, long index){
    if (!px){return;} 
    for (size_t x(0); x < px -> mass.size(); ++x){
        
        rectop rtx; 
        rtx.decode(px -> chn[x]);
        rtx.mass  = px -> mass[x] * 0.001;
        rtx.pt    = px -> pt[x]   * 0.001; 
        rtx.eta   = px -> eta[x]; 
        rtx.phi   = px -> phi[x];
        rtx.index = index; 

        if (px -> type == "tops_truth"){
            rtx.idt = object_e::truth;
            ev -> truth.push_back(rtx);
            continue;
        }

        rtx.score = px -> score[x]; 
        if (px -> type == "tops_nom"){ev -> nominal.push_back(rtx); continue;}
        if (px -> type == "tops_upr"){ev -> umasked.push_back(rtx); continue;}
        if (px -> type == "tops_pr" ){ev -> masked.push_back(rtx);  continue;}
    }
}

int collector::inlet(edata* data){

    std::string mode = ""; 
    mode += this -> label(&data -> training  , "training"  ); 
    mode += this -> label(&data -> validation, "validation"); 
    mode += this -> label(&data -> evaluation, "evaluation"); 
  
    std::map<std::string, performance*>* prf = nullptr;  
    if      (mode == "training"  ){prf = &this -> training;  }
    else if (mode == "validation"){prf = &this -> validation;}
    else if (mode == "evaluation"){prf = &this -> evaluation;}
    else {abort();}

    std::string key = data -> mrk; 
    if (!prf -> count(key)){(*prf)[key] = new performance(key, mode);}
    
    edata_t ev = edata_t();
    ev.ntru = data -> ntop_tru;
    ev.npre = data -> ntop_prd;
    ev.pred = data -> ntop_score; 

    ev.acc_edges = data -> acc_edge;
    ev.prc = process_sample(nullptr, &data -> dsid); 
    for (size_t x(0); x < data -> training.size();   ++x){expand(&ev, data -> training[x]  , data -> index);}
    for (size_t x(0); x < data -> validation.size(); ++x){expand(&ev, data -> validation[x], data -> index);}
    for (size_t x(0); x < data -> evaluation.size(); ++x){expand(&ev, data -> evaluation[x], data -> index);}
    int _epc = data -> epoch;
    int _epk = data -> kfold;  

    ev.truth.shrink_to_fit();
    ev.nominal.shrink_to_fit();
    ev.umasked.shrink_to_fit();
    ev.masked.shrink_to_fit(); 

    performance* pr = (*prf)[key];
    if (this -> epoch < 0){this -> epoch = _epc;}
    if (this -> kfold < 0){this -> kfold = _epk;}

    bool epx = this -> epoch != _epc;
    bool kpx = this -> kfold != _epk;
    if (epx){this -> epoch = _epc;}
    if (kpx){this -> kfold = _epk;}
    pr -> epoch[_epc][_epk].push_back(std::move(ev)); 
    return pr -> epoch[_epc][_epk].size(); 
}

