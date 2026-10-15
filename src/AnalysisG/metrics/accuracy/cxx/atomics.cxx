#include <metrics/atomics.h>

atomic_t::atomic_t(){}
atomic_t::~atomic_t(){}

long atomic_t::add_truth(double m){
    if (m < 0){return 1;}
    this -> truth.push_back(m); 
    return 1; 
}

long atomic_t::add_value(double m){
    if (m < 0){return 1;}
    this -> masses.push_back(m); 
    return 1; 
}

long atomic_t::add_weight(double m){
    this -> weight.push_back(m); 
    return 1; 
}



pdax_t::pdax_t(){}

pdax_t::~pdax_t(){
    this -> pflush(&this -> mlp); 
    this -> pflush(&this -> weight);
    this -> pflush(&this -> masses); 
    this -> pflush(&this -> ntrus); 
    this -> pflush(&this -> ntops);
    this -> pflush(&this -> index); 
}

void pdax_t::dmake(pdax_t* ptr){
    long sx = this -> masses -> size();
    long sy = ptr  -> masses -> size();
    long mx = (sx > sy) ? sx : sy; 
    pdax_t* sv = (sx < sy) ? ptr  : this; 
    pdax_t* sj = (sx < sy) ? this :  ptr; 

    std::vector<long>   idx = std::vector<long>(mx, -1); 
    std::vector<double> dSo = std::vector<double>(mx, -1);
    std::vector<double>* Sv = sv -> masses;
    std::vector<double>* Sj = sj -> masses; 

    long kx = 0; long dy = 0;
    sy = Sv -> size(); sx = Sj -> size();
    while (true){
        long ky = kx + dy; 
        double lx = (kx >= sx) ? -1.0 : Sj -> at(kx); 
        double ly = (ky >= sy) ? -1.0 : Sv -> at(ky);
        bool sl = (ly == lx) * (lx > -1.0) * (ly > -1.0); 
        if (lx < 0 && ly < 0){break;}
        dSo[ky] = (sl) ? ly : -1;
        idx[ky] = kx;
        dy += (!sl); kx += sl;
    }

//    Sj -> clear(); 
//    for (size_t x(0); x < mx; ++x){Sj -> push_back(dSo[x]);}
}
