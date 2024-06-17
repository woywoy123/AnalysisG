#ifndef NOTIFICATION_NOTIFICATION_H
#define NOTIFICATION_NOTIFICATION_H
#include <string>
#include <iostream>
#include <sstream>

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
}; 


#endif
