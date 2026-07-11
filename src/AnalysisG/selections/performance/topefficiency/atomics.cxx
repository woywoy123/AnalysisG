#include "topefficiency.h"

selection_template* topefficiency::clone(){return (selection_template*)new topefficiency();}
bool topefficiency::selection(event_template* ev){return true;}
void topefficiency::check_matching(dump_t* ptx, top* trg, std::vector<top*>* ev_tops){
    double tpx = trg -> px; 
    double tpy = trg -> py; 
    double tpz = trg -> pz; 
    double tms = trg -> mass; 

    long bst = 0; 
    double lowC = -1; 
    for (size_t x(0); x < ev_tops -> size(); ++x){
        double px = ev_tops -> at(x) -> px; 
        double py = ev_tops -> at(x) -> py; 
        double pz = ev_tops -> at(x) -> pz; 
        double ms = ev_tops -> at(x) -> mass; 

        double dx = (px - tpx) * (px - tpx); 
        double dy = (py - tpy) * (py - tpy); 
        double dz = (pz - tpz) * (pz - tpz); 
        double dm = (ms - tms) * (ms - tms); 

        double sm = dx + dy + dz + dm; 
        if (lowC < 0){lowC = sm;}
        if (lowC < sm){continue;}
        lowC = sm; bst = x;  
    }
    if (lowC < 0){return;}
    top* tx = ev_tops -> at(bst); 
    ptx -> transfer(ev_tops -> at(bst)); 
    ptx -> lowC.push_back(lowC); 
    if (!tx -> av_score){return;}
    ptx -> ranks.push_back(tx -> av_score); 
}





