#include <modules/notification.h>
#include <tools/tools.h>

void AnalysisG::modules::notification::monitor(std::vector<std::thread*>* thr){
    size_t exec = thr -> size(); 
    while (exec){
        exec = thr -> size(); 
        for (size_t x(0); x < thr -> size(); ++x){
            if (!(*thr)[x]){--exec; continue;}
            if (!(*thr)[x] -> joinable()){continue;}
            (*thr)[x] -> join(); 
            AnalysisG::tooling::pflush(&(*thr)[x]); 
            --exec; 
        }
    }
}


int AnalysisG::modules::notification::running(std::vector<std::thread*>* thr, std::vector<size_t>* prg, std::vector<size_t>* trgt){
    size_t idx = 0; 
    for (size_t x(0); x < thr -> size(); ++x){
        if (!(*thr)[x]){continue;}
        if (!(*trgt)[x] && !(*prg)[x]){continue;}
        if (!(*thr)[x] -> joinable()){++idx; continue;}
        if ((*trgt)[x] != (*prg)[x]){++idx; continue;}
        (*thr)[x] -> join(); 
        AnalysisG::tooling::pflush(&(*thr)[x]); 
    }
    return int(idx); 
}


