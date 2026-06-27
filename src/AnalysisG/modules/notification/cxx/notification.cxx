#include <notification/notification.h>
#include <stddef.h>
#include <iomanip>

multithreaded_t::multithreaded_t(size_t lgt, int num_thr){
    this -> progress = new std::vector<size_t>(lgt, 0); 
    this -> target   = new std::vector<size_t>(lgt, 0); 
    this -> status   = new std::vector<size_t>(lgt, 1); 
    this -> threads  = new std::vector<std::thread*>(lgt, nullptr);
    this -> coms     = new std::vector<std::string*>(lgt, nullptr); 
    this -> traces   = new std::vector<tracing_t*>(lgt, nullptr); 
    for (size_t x(0); x < lgt; ++x){
        (*this -> coms)[x]   = new std::string("");

        (*this -> traces)[x] = new tracing_t(); 
        (*this -> traces)[x] -> coms      =  (*this -> coms)[x]; 
        (*this -> traces)[x] -> idx       = &(*this -> progress)[x]; 
        (*this -> traces)[x] -> status    = &(*this -> status)[x]; 
        (*this -> traces)[x] -> maxlength = &(*this -> target)[x]; 
        (*this -> traces)[x] -> threadIdx = x; 
        (*this -> traces)[x] -> reg       = this; 
    }
    this -> num_threads = num_thr;
    this -> job_length = lgt; 
}

multithreaded_t::~multithreaded_t(){
    this -> vflush(this -> threads); 
    this -> vflush(this -> traces);

    this -> pflush(&this -> status); 
    this -> pflush(&this -> progress); 
    this -> pflush(&this -> threads); 
    this -> pflush(&this -> target); 
    this -> pflush(&this -> traces); 
    this -> pflush(&this -> coms); 
}

size_t tracing_t::index(){return (*this -> idx);}
void   tracing_t::next(){(*this -> idx)++;}
void   tracing_t::finished(){(*this -> status) = 0;}

void   tracing_t::message(std::string msg){(*this -> coms) = msg;}
void   tracing_t::register_thread(std::thread* thr, size_t x){
    (*this -> reg -> threads)[this -> threadIdx] = thr;
    (*this -> reg -> target )[this -> threadIdx] = x; 
    (*this -> reg -> status )[this -> threadIdx] = 1; 
    thr -> detach(); 
}




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

    for (size_t x(0); x < bars.size(); ++x){
        if (!bars[x]){continue;}
        delete bars[x];
    }
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

int notification::running(std::vector<std::thread*>* thr, std::vector<size_t>* prg, std::vector<size_t>* trgt){
    size_t idx = 0; 
    for (size_t x(0); x < thr -> size(); ++x){
        if (!(*thr)[x]){continue;}
        if (!(*trgt)[x] && !(*prg)[x]){continue;}
        if (!(*thr)[x] -> joinable()){++idx; continue;}
        if ((*trgt)[x] != (*prg)[x]){++idx; continue;}
        (*thr)[x] -> join(); 
        delete (*thr)[x]; 
        (*thr)[x] = nullptr; 
    }
    return int(idx); 
}

multithreaded_t* notification::make_threads(size_t num_jobs, int num_threads){
    multithreaded_t* thr = new multithreaded_t(num_jobs, num_threads); 
    if (this -> shush){return thr;}
    thr -> ptr = new std::thread(this -> progressbar3, thr -> progress, thr -> target, thr -> coms); 
    return thr;   
}

bool notification::await_threads(multithreaded_t* thr, bool monitor){
    int cnt = 0; 
    for (size_t x(0); x < thr -> job_length; ++x){
        tracing_t* tr = thr -> traces -> at(x); 
        if (!(*thr -> status )[x]){continue;}
        if (!(*thr -> threads)[x]){continue;}
        ++cnt; 
    }
    if (cnt > thr -> num_threads && !monitor){return true;}
    if (monitor && cnt > 0){return true;}

    if (!monitor){return false;} 
    if (!thr -> ptr){return false;}
    thr -> ptr -> join(); delete thr -> ptr; thr -> ptr = nullptr;   
    return false; 
}

