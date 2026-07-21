#ifndef ANALYSISG_STRUCTS_MULTITHREAD_H
#define ANALYSISG_STRUCTS_MULTITHREAD_H
#include <tools/tools.h>
#include <vector>
#include <thread>
#include <string>

namespace AnalysisG {
    namespace core {
        struct tracing_t; 

        struct multithreaded_t {
            multithreaded_t(size_t lgt, int num_thr); 
            ~multithreaded_t();
        
            std::vector<AnalysisG::core::tracing_t*>* traces = nullptr; 
            std::vector<size_t>*       status   = nullptr;  
            std::vector<size_t>*       progress = nullptr; 
            std::vector<size_t>*       target   = nullptr;  
            std::vector<std::thread*>* threads  = nullptr;  
            std::vector<std::string*>* coms     = nullptr;  
        
            int num_threads    = -1; 
            size_t job_length  = 0; 
            std::thread* ptr   = nullptr; 
        }; 
    }
}
#endif
