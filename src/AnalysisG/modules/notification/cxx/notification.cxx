#include <modules/notification.h>
#include <tools/tools.h>

AnalysisG::modules::notification::notification(){}
AnalysisG::modules::notification::~notification(){}

void AnalysisG::modules::notification::success(std::string message){
    this -> caller = this -> _success;
    this -> _format(&message);
}

void AnalysisG::modules::notification::warning(std::string message){
    this -> caller = this -> _warning;
    bool s = this -> shush; 
    this -> shush = false; 
    this -> _format(&message);
    this -> shush = s; 
}

void AnalysisG::modules::notification::failure(std::string message){
    this -> caller = this -> _failure;
    bool s = this -> shush; 
    this -> shush = false; 
    this -> _format(&message);
    this -> shush = s; 
}

void AnalysisG::modules::notification::info(std::string message){
    this -> caller = this -> _info;
    this -> _format(&message);
}


AnalysisG::core::multithreaded_t* AnalysisG::modules::notification::make_threads(size_t num_jobs, int num_threads){
    AnalysisG::core::multithreaded_t* thr = new AnalysisG::core::multithreaded_t(num_jobs, num_threads); 
    if (this -> shush){return thr;}
    thr -> ptr = new std::thread(this -> progressbar3, thr -> progress, thr -> target, thr -> coms); 
    return thr;   
}

bool AnalysisG::modules::notification::await_threads(AnalysisG::core::multithreaded_t* thr, bool monitor){
    int cnt = 0; 
    for (size_t x(0); x < thr -> job_length; ++x){
        if (!(*thr -> status )[x]){continue;}
        if (!(*thr -> threads)[x]){continue;}
        ++cnt; 
    }

    if (cnt > thr -> num_threads && !monitor){return true;}
    if (monitor && cnt > 0){return true;}

    if (!monitor){return false;} 
    if (!thr -> ptr){return false;}

    thr -> ptr -> join();
    AnalysisG::tooling::pflush(&thr -> ptr); 
    return false; 
}


