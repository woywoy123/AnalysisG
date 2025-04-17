#ifndef NOTIFICATION_NOTIFICATION_H
#define NOTIFICATION_NOTIFICATION_H

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>

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
