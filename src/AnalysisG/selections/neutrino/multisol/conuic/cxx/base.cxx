#include <conuic/atomics.h>
#include <conuic/base.h>
#include <math.h>

debug_t::debug_t(){}
debug_t::~debug_t(){
    std::map<std::string, long double*>::iterator it = this -> dlt.begin();
    for (; it != this -> dlt.end(); ++it){flush(&it -> second);}
    this -> dlt.clear();
}

void debug_t::track(std::string name, long double* val){this -> trk[name] = val;}

void debug_t::_track(std::string name, long double* val){
    this -> trk[name] = val; this -> dlt[name] = val; 
}

std::string debug_t::print(bool cx, int prec){
    std::map<std::string, long double*>::iterator it = this -> trk.begin();
    std::string out = ""; 
    for (; it != this -> trk.end(); ++it){
        if (!it -> second){continue;}
        out += it -> first + " -> "; 
        out += tools::to_string(*it -> second, prec) + "\n"; 
    }
    if (!cx){return out;}
    std::cout << out << std::endl;
    return out; 
}

bool debug_t::assertions(std::string name, long double t1, long double v1, long double tol){
    long double v = (t1 - v1) / (t1 + (t1 == 0) * 1.0L); 
    bool eq = (std::fabs(v) < tol); 
    name = ((eq) ? "OK! " : "!!!! ERROR !!!!! ") + name ; 
    this -> _track("V -> " + name, new long double(v1)); 
    this -> _track("D -> " + name, new long double(v)); 
    if (!eq){ this -> _track("T -> " + name, new long double(t1)); }
    return std::fabs(v) < tol; 
}

void debug_t::print(std::string name, int prec){
    if (!this -> trk.count(name)){return;}
    std::cout << name + " +> \n";
    std::cout << tools::to_string(*this -> trk[name], prec) + "\n";
    std::cout << std::endl; 
}

kinematics_t::~kinematics_t(){
    this -> ptr_ = nullptr; 
}

kinematics_t::kinematics_t(){
    this -> track("px", &this -> px); 
    this -> track("py", &this -> py);
    this -> track("pz", &this -> pz);
    this -> track("e" , &this -> e);
    this -> track("mass", &this -> m); 
}

kinematics_t::kinematics_t(particle_template* ptr){
    this -> px = convert(ptr -> px);
    this -> py = convert(ptr -> py);
    this -> pz = convert(ptr -> pz);

    this -> e  = convert(ptr -> e   ); 
    this -> m  = convert(ptr -> mass);
    this -> b  = convert(ptr -> beta);
    this -> p  = convert(ptr -> P); 

    this -> track("px", &this -> px); 
    this -> track("py", &this -> py);
    this -> track("pz", &this -> pz);
    this -> track("e" , &this -> e);
    this -> track("mass", &this -> m); 
    this -> ptr_ = ptr; 
}




