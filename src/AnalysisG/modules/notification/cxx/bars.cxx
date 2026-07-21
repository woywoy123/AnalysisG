#include <modules/notification.h>
#include <tools/tools.h>
#include <sstream>
#include <stddef.h>
#include <iostream>
#include <iomanip>

void AnalysisG::modules::notification::progressbar1(std::vector<size_t>* threads, size_t l, std::string title){
    AnalysisG::modules::notification ntx = AnalysisG::modules::notification(); 
    while (true){
        std::this_thread::sleep_for(std::chrono::milliseconds(ntx.refresh));
        float prgs = float(AnalysisG::tooling::sum(threads))/float(l); 
        ntx.progressbar(prgs, title); 
        if (prgs > 0.995){break;}
    } 
    ntx.progressbar(1, title); 
    std::cout << "" << std::endl;
} 

void AnalysisG::modules::notification::progressbar2(std::vector<size_t>* threads, size_t* l, std::string* title){
    AnalysisG::modules::notification ntx = AnalysisG::modules::notification(); 
    while (true){
        std::this_thread::sleep_for(std::chrono::milliseconds(ntx.refresh));
        float prgs = float(AnalysisG::tooling::sum(threads))/float(*l); 
        ntx.progressbar(prgs, *title); 
        if (prgs > 0.995){break;}
    } 
    ntx.progressbar(1, *title); 
    std::cout << "" << std::endl;
} 



void AnalysisG::modules::notification::progressbar3(std::vector<size_t>* threads, std::vector<size_t>* l, std::vector<std::string*>* title){
    if (!title){return;}

    AnalysisG::modules::notification ntx = AnalysisG::modules::notification();
    std::vector<std::string*> bars(l -> size(), nullptr); 
    for (size_t x(0); x < l -> size(); ++x){
        std::string* bi = nullptr; 
        if (title && (*title)[x]){bi = (*title)[x];}
        else {bi = new std::string("Thread (" + std::to_string(x+1) + ")");}
        bars[x] = bi; 
    }

    float prgs = 0; 
    size_t cwhite = 0; 
    while (prgs < 1.0){
        std::this_thread::sleep_for(std::chrono::milliseconds(ntx.refresh));
        size_t xl = AnalysisG::tooling::sum(l);  
        size_t xp = AnalysisG::tooling::sum(threads);
        if (!xl){continue;}

        prgs = float(xp)/float(xl); 
        std::vector<size_t> prx = {}; 
        std::vector<size_t> totl = {}; 
        std::vector<std::string> vec = {};
        size_t ln = 0; 

        for (size_t x(0); x < bars.size(); ++x){
            if (!(*threads)[x]){continue;}
            if ((*threads)[x] == (*l)[x]){continue;}
            vec.push_back(std::string(bars[x] -> c_str()));
            prx.push_back((*threads)[x]);
            totl.push_back((*l)[x]); 
            ln = (bars[x] -> size() < ln) ? ln : bars[x] -> size(); 
        }

        prx.push_back(xp); 
        totl.push_back(xl); 
        vec.push_back("Total Progress:"); 

        for (size_t x(0); x < prx.size()-1; ++x){
            size_t t = vec[x].size(); 
            for (size_t y(t); y < ln; ++y){vec[x]+= " ";}
        }

        for (size_t x(0); x < cwhite; ++x){std::cout << "\033[F\x1b[2K";}
        std::cout << std::flush; 
        ntx.progressbar(&prx, &totl, &vec);  
        cwhite = prx.size();
    }
    AnalysisG::tooling::vflush(&bars); 
} 




