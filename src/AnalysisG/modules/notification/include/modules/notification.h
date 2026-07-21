#ifndef ANALYSISG_MODULES_NOTIFICATION_H
#define ANALYSISG_MODULES_NOTIFICATION_H

#include <string>
#include <vector>
#include <thread>

#include <structs/tracing.h>
#include <structs/multithreaded.h>

namespace AnalysisG {
    namespace modules {
        class notification {
            public:
                notification(); 
                ~notification(); 
        
                void success(std::string message); 
                void warning(std::string message);
                void failure(std::string message);
                void info(std::string message);

                void progressbar(float prog, std::string title); 
                void progressbar(std::vector<size_t>* threads, std::vector<size_t>* trgt, std::vector<std::string>* title); 

                void static progressbar1(std::vector<size_t>* threads, size_t l, std::string title); 
                void static progressbar2(std::vector<size_t>* threads, size_t* l, std::string* title); 
                void static progressbar3(std::vector<size_t>* threads, std::vector<size_t>* l, std::vector<std::string*>* title); 
 
                AnalysisG::core::multithreaded_t* make_threads(size_t num_jobs, int num_threads);  
                bool await_threads(AnalysisG::core::multithreaded_t* thr, bool monitor); 
     
                int  running(std::vector<std::thread*>* thr, std::vector<size_t>* prg, std::vector<size_t>* trgt); 
                void monitor(std::vector<std::thread*>* thr); 
                
                int  refresh = 10; 
                bool shush = false; 
                std::string prefix; 
        
            private:
                void _format(std::string* message); 

                bool _bold = false; 
                int _warning = 33;
                int _failure = 31; 
                int _success = 32;
                int _info    = 37; 
                int caller;
        }; 
    }
}


#endif
