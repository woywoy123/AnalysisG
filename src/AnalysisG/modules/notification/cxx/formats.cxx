#include <modules/notification.h>

#include <sstream>
#include <stddef.h>
#include <iostream>
#include <iomanip>

void AnalysisG::modules::notification::_format(std::string* message){
    if (this -> shush){return;}
    std::stringstream stream; 
    stream << "\033["; 
    if (this -> _bold){ stream << "1;"; }
    stream << this -> caller; 
    stream << "m"; 
    if (this -> prefix.size()){stream << this -> prefix << "::";}
    stream << *message; 
    stream << "\033[0m"; 
    std::cout << stream.str() << std::endl; 
}

void AnalysisG::modules::notification::progressbar(float lProgress, std::string title){
    const char cFilled[] = "#####################################";
    const char cEmpty[]  = "                                     ";
    size_t lFilledStart = (sizeof cFilled - 1) * (1 - lProgress);
    size_t lEmptyStart  = (sizeof cFilled - 1) * lProgress;
    printf("\r %s | [%s%s] %.1f%%", title.c_str(), cFilled + lFilledStart, cEmpty + lEmptyStart, lProgress * 100);
    fflush(stdout);
}

void AnalysisG::modules::notification::progressbar(std::vector<size_t>* threads, std::vector<size_t>* trgt, std::vector<std::string>* title){
    const char cFilled[] = "#####################################";
    const char cEmpty[]  = "                                     ";
    for (size_t x(0); x < trgt -> size(); ++x){
        float prg = float(threads -> at(x)) / float(trgt -> at(x)); 
        size_t lFilledStart = (sizeof(cFilled) - 1) * (1 - prg);
        size_t lEmptyStart  = (sizeof(cFilled) - 1) * prg;
        std::cout << title -> at(x) << " [" << cFilled + lFilledStart << cEmpty + lEmptyStart << "] ";  
        std::cout << std::fixed << std::setprecision(4) << prg*100 << "%\n"; 
    }
    std::cout << std::flush; 
}


