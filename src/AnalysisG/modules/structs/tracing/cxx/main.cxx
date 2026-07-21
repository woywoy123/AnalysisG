#include <structs/multithreaded.h>
#include <structs/tracing.h>

size_t AnalysisG::core::tracing_t::index(){return (*this -> idx);}
void   AnalysisG::core::tracing_t::next(){(*this -> idx)++;}
void   AnalysisG::core::tracing_t::finished(){(*this -> status) = 0;}

void   AnalysisG::core::tracing_t::message(std::string msg){(*this -> coms) = msg;}
void   AnalysisG::core::tracing_t::register_thread(std::thread* thr, size_t x){
    (*this -> reg -> threads)[this -> threadIdx] = thr;
    (*this -> reg -> target )[this -> threadIdx] = x; 
    (*this -> reg -> status )[this -> threadIdx] = 1; 
    thr -> detach(); 
}




