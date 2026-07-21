#ifndef ANALYSISG_STRUCTS_TRACING_H
#define ANALYSISG_STRUCTS_TRACING_H

#include <tools/tools.h>
#include <vector>
#include <thread>
#include <string>


namespace AnalysisG {
    namespace core {
        struct multithreaded_t; 

        struct tracing_t {
            void next(); 
            void finished(); 
            void register_thread(std::thread* thr, size_t x); 
            void message(std::string msg); 
            size_t index();
         
            size_t     threadIdx = 0;   
            size_t*    maxlength = nullptr; 
            size_t*       status = nullptr; 
            size_t*          idx = nullptr; 

            std::string*    coms = nullptr; 
            AnalysisG::core::multithreaded_t* reg = nullptr; 
        };
    }
}
   
#endif
