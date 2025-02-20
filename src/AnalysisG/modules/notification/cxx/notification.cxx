#include "notification.h"
#include <stddef.h>
#include <iomanip>

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
    const char cFilled[] = "|||||||||||||||||||||||||||||||||||||";
    const char cEmpty[]  = "                                     ";
    size_t lFilledStart = (sizeof cFilled - 1) * (1 - lProgress);
    size_t lEmptyStart  = (sizeof cFilled - 1) * lProgress;
    printf("\r %s | [%s%s] %.1f%%", title.c_str(), cFilled + lFilledStart, cEmpty + lEmptyStart, lProgress * 100);
    fflush(stdout);
}

void notification::progressbar(std::vector<size_t>* threads, std::vector<size_t>* trgt, std::vector<std::string>* title){
    const char cFilled[] = "|||||||||||||||||||||||||||||||||||||";
    const char cEmpty[]  = "                                     ";
    for (size_t x(0); x < trgt -> size(); ++x){std::cout << "\033[F";}
    for (size_t x(0); x < trgt -> size(); ++x){
        float prg = float(threads -> at(x)) / float(trgt -> at(x)); 
        size_t lFilledStart = (sizeof(cFilled) - 1) * (1 - prg);
        size_t lEmptyStart  = (sizeof(cFilled) - 1) * prg;
        std::cout << title -> at(x) << " [" << cFilled + lFilledStart << cEmpty + lEmptyStart << "] ";  
        std::cout << std::fixed << std::setprecision(4) << prg*100 << "%\n"; 
    }
    std::cout << std::flush; 
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
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        float prgs = float(notification().sum(threads))/float(*l); 
        if (prgs > 0.995){break;}
        notification().progressbar(prgs, *title); 
    } 
    notification().progressbar(1, *title); 
    std::cout << "" << std::endl;
} 


void notification::progressbar3(std::vector<size_t>* threads, std::vector<size_t>* l, std::vector<std::string*>* title){
    notification n = notification(); 
    for (size_t x(0); x < l -> size(); ++x){
        if (!(*title)[x]){(*title)[x] = new std::string("");}
        std::cout << "" << std::endl;
    }
    while (true){
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        float prgs = float(n.sum(threads))/float(n.sum(l)); 
        if (prgs >= 1){break;}
        std::vector<std::string> vec = {};
        for (size_t x(0); x < title -> size(); ++x){vec.push_back(std::string((*title)[x] -> c_str()));}
        n.progressbar(threads, l, &vec);  
    } 
    for (size_t x(0); x < title -> size(); ++x){delete (*title)[x];}
    std::cout << "" << std::endl;
} 

void notification::monitor(std::vector<std::thread*>* thr){
    for (size_t x(0); x < thr -> size(); ++x){
        if (!thr -> at(x)){continue;}
        if (!thr -> at(x) -> joinable()){continue;}
        (*thr)[x] -> join(); 
        delete (*thr)[x]; 
        (*thr)[x] = nullptr; 
        x = 0; 
    }
}

int notification::running(std::vector<std::thread*>* thr){
    int idx = 0; 
    for (size_t x(0); x < thr -> size(); ++x){
        if (!thr -> at(x)){continue;}
        if (!thr -> at(x) -> joinable()){idx++; continue;}
        (*thr)[x] -> join(); 
        delete (*thr)[x]; 
        (*thr)[x] = nullptr; 
        --idx; 
    }
    return idx; 
}


