#include <structs/multithreaded.h>
#include <structs/tracing.h>

AnalysisG::core::multithreaded_t::multithreaded_t(size_t lgt, int num_thr){
    this -> progress = new std::vector<size_t>(lgt, 0); 
    this -> target   = new std::vector<size_t>(lgt, 0); 
    this -> status   = new std::vector<size_t>(lgt, 1); 
    this -> threads  = new std::vector<std::thread*>(lgt, nullptr);
    this -> coms     = new std::vector<std::string*>(lgt, nullptr); 
    this -> traces   = new std::vector<AnalysisG::core::tracing_t*>(lgt, nullptr); 
    for (size_t x(0); x < lgt; ++x){
        (*this -> coms)[x]   = new std::string("");

        (*this -> traces)[x] = new AnalysisG::core::tracing_t(); 
        (*this -> traces)[x] -> coms      =  (*this -> coms)[x]; 
        (*this -> traces)[x] -> idx       = &(*this -> progress)[x]; 
        (*this -> traces)[x] -> status    = &(*this -> status)[x]; 
        (*this -> traces)[x] -> maxlength = &(*this -> target)[x]; 
        (*this -> traces)[x] -> threadIdx = x; 
        (*this -> traces)[x] -> reg       = this; 
    }
    this -> num_threads = num_thr;
    this -> job_length = lgt; 
}

AnalysisG::core::multithreaded_t::~multithreaded_t(){
    AnalysisG::tooling::vflush(this -> threads); 
    AnalysisG::tooling::vflush(this -> traces);

    AnalysisG::tooling::pflush(&this -> status); 
    AnalysisG::tooling::pflush(&this -> progress); 
    AnalysisG::tooling::pflush(&this -> threads); 
    AnalysisG::tooling::pflush(&this -> target); 
    AnalysisG::tooling::pflush(&this -> traces); 
    AnalysisG::tooling::pflush(&this -> coms); 
}


