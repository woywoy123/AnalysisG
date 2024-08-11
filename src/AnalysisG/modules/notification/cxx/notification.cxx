#include "notification.h"
#include <stddef.h>
#include <thread>

notification::notification(){}
notification::~notification(){}

void notification::_format(std::string* message){
    if (this -> shush){return;}
    std::stringstream stream; 
    stream << "\033["; 
    if (this -> bold){ stream << "1;"; }
    stream << this -> caller; 
    stream << "m"; 
    if (this -> prefix.size()){ 
        stream << this -> prefix << "::"; 
    }
    stream << *message; 
    stream << "\033[0m"; 
    std::cout << stream.str() << std::endl; 
}

void notification::success(std::string message){
    this -> caller = this -> _success;
    this -> _format(&message);
}

void notification::warning(std::string message){
    this -> caller = this -> _warning;
    bool s = this -> shush; 
    this -> shush = false; 
    this -> _format(&message);
    this -> shush = s; 
}

void notification::failure(std::string message){
    this -> caller = this -> _failure;
    bool s = this -> shush; 
    this -> shush = false; 
    this -> _format(&message);
    this -> shush = s; 
}

void notification::info(std::string message){
    this -> caller = this -> _info;
    this -> _format(&message);
}

void notification::progressbar(float lProgress, std::string title){
    const char cFilled[] = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX";
    const char cEmpty[]  = "-------------------------------------";
    size_t lFilledStart = (sizeof cFilled - 1) * (1 - lProgress);
    size_t lEmptyStart  = (sizeof cFilled - 1) * lProgress;
    printf("\r %s | [%s%s] %.1f%%", title.c_str(), cFilled + lFilledStart, cEmpty  + lEmptyStart, lProgress * 100);
    fflush(stdout);
}

void notification::progressbar1(std::vector<size_t>* threads, size_t l, std::string title){
    while (true){
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        float prgs = float(notification().sum(threads))/float(l); 
        if (prgs > 0.995){break;}
        notification().progressbar(prgs, title); 
    } 
    notification().progressbar(1, title); 
    std::cout << "" << std::endl;
} 

void notification::progressbar2(std::vector<size_t>* threads, size_t* l, std::string* title){
    while (true){
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        float prgs = float(notification().sum(threads))/float(*l); 
        if (prgs > 0.995){break;}
        notification().progressbar(prgs, *title); 
    } 
    notification().progressbar(1, *title); 
    std::cout << "" << std::endl;
} 






