#ifndef NOTIFICATION_NOTIFICATION_H
#define NOTIFICATION_NOTIFICATION_H

#include <tools/tools.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>

class tools; 
class notification; 
struct multithreaded_t; 

struct tracing_t {
    void next(); 
    void finished(); 
    void register_thread(std::thread* thr, size_t x); 
    void message(std::string msg); 
    size_t index();
 
    size_t     threadIdx = 0;   
    size_t*       status = nullptr; 
    size_t*          idx = nullptr; 
    size_t*    maxlength = nullptr; 
    std::string*    coms = nullptr; 
    multithreaded_t* reg = nullptr; 
};


struct multithreaded_t : public tools {
    multithreaded_t(size_t lgt, int num_thr); 
    ~multithreaded_t();

    std::vector<size_t>*       status   = nullptr;  
    std::vector<size_t>*       progress = nullptr; 
    std::vector<size_t>*       target   = nullptr;  
    std::vector<std::thread*>* threads  = nullptr;  
    std::vector<std::string*>* coms     = nullptr;  
    std::vector<tracing_t*>*   traces   = nullptr; 

    int num_threads    = -1; 
    size_t job_length  = 0; 
    std::thread* ptr   = nullptr; 
}; 




class notification
{
    public:
        notification(); 
        ~notification(); 

        void success(std::string message); 
        void warning(std::string message);
        void failure(std::string message);
        void info(std::string message);
        void progressbar(float prog, std::string title); 
        void progressbar(std::vector<size_t>* threads, std::vector<size_t>* trgt, std::vector<std::string>* title); 

        int  running(std::vector<std::thread*>* thr, std::vector<size_t>* prg, std::vector<size_t>* trgt); 
        void monitor(std::vector<std::thread*>* thr); 
        
        multithreaded_t* make_threads(size_t num_jobs, int num_threads);  
        bool await_threads(multithreaded_t* thr, bool monitor); 

        void static progressbar1(std::vector<size_t>* threads, size_t l, std::string title); 
        void static progressbar2(std::vector<size_t>* threads, size_t* l, std::string* title); 
        void static progressbar3(std::vector<size_t>* threads, std::vector<size_t>* l, std::vector<std::string*>* title); 

        std::string prefix; 
        int _warning = 33;
        int _failure = 31; 
        int _success = 32;
        int _info = 37; 
        bool bold = false; 
        bool shush = false; 

    private:
        void _format(std::string* message); 
        int caller;

        template <typename g>
        g sum(std::vector<g>* inpt){
            g ix = 0; 
            for (size_t t(0); t < inpt -> size(); ++t){ix += (*inpt)[t];}
            return ix; 
        }
}; 


#endif
