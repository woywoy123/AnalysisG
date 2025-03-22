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
    if (this -> prefix.size()){stream << this -> prefix << "::";}
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
    const char cFilled[] = "#####################################";
    const char cEmpty[]  = "                                     ";
    size_t lFilledStart = (sizeof cFilled - 1) * (1 - lProgress);
    size_t lEmptyStart  = (sizeof cFilled - 1) * lProgress;
    printf("\r %s | [%s%s] %.1f%%", title.c_str(), cFilled + lFilledStart, cEmpty + lEmptyStart, lProgress * 100);
    fflush(stdout);
}

void notification::progressbar(std::vector<size_t>* threads, std::vector<size_t>* trgt, std::vector<std::string>* title){
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

void notification::progressbar1(std::vector<size_t>* threads, size_t l, std::string title){
    while (true){
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        float prgs = float(notification().sum(threads))/float(l); 
        notification().progressbar(prgs, title); 
        if (prgs > 0.995){break;}
    } 
    notification().progressbar(1, title); 
    std::cout << "" << std::endl;
} 

void notification::progressbar2(std::vector<size_t>* threads, size_t* l, std::string* title){
    while (true){
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        float prgs = float(notification().sum(threads))/float(*l); 
        notification().progressbar(prgs, *title); 
        if (prgs > 0.995){break;}
    } 
    notification().progressbar(1, *title); 
    std::cout << "" << std::endl;
} 


void notification::progressbar3(std::vector<size_t>* threads, std::vector<size_t>* l, std::vector<std::string*>* title){
    notification n = notification();
    if (!title){return;}
    std::vector<std::string*> bars = {}; 
    for (size_t x(0); x < l -> size(); ++x){
        std::string* bi = nullptr; 
        if (title && (*title)[x]){bi = (*title)[x];}
        else {bi = new std::string("Thread (" + std::to_string(x+1) + ")");}
        bars.push_back(bi); 
    }

    float prgs = 0; 
    size_t cwhite = 0; 
    while (prgs < 1.0){
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        size_t xl = n.sum(l);  
        size_t xp = n.sum(threads);
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
        n.progressbar(&prx, &totl, &vec);  
        cwhite = prx.size();
    }

    for (size_t x(0); x < bars.size(); ++x){delete bars[x];}
} 

void notification::monitor(std::vector<std::thread*>* thr){
    size_t exec = thr -> size(); 
    while (exec){
        exec = thr -> size(); 
        for (size_t x(0); x < thr -> size(); ++x){
            if (!(*thr)[x]){--exec; continue;}
            if (!(*thr)[x] -> joinable()){continue;}
            (*thr)[x] -> join(); 
            delete (*thr)[x]; 
            (*thr)[x] = nullptr; 
            --exec; 
        }
    }
}

int notification::running(std::vector<std::thread*>* thr){
    size_t idx = thr -> size(); 
    for (size_t x(0); x < thr -> size(); ++x){
        if (!(*thr)[x]){--idx; continue;}
        if (!(*thr)[x] -> joinable()){continue;}
        (*thr)[x] -> join(); 
        delete (*thr)[x]; 
        (*thr)[x] = nullptr; 
        --idx; 
    }
    return int(idx); 
}


