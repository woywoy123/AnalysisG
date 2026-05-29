#ifndef MULTISOL_CONUIC_BASE_H
#define MULTISOL_CONUIC_BASE_H

#include <templates/particle_template.h>
#include <tools/tools.h>

struct debug_t {
    debug_t();
    ~debug_t(); 
    void track(std::string name, long double* val); 
    void _track(std::string name, long double* val); 

    void print(std::string name, int prec = 12);    
    std::string print(bool cx = true, int prec = 12);

    bool assertions(std::string name, long double t1, long double v1, long double tol = 1e-3); 

    std::map<std::string, long double*> trk = {}; 
    std::map<std::string, long double*> dlt = {}; 
}; 

struct kinematics_t : public debug_t {
    kinematics_t(particle_template* ptr); 
    kinematics_t(); 

    ~kinematics_t(); 

    long double px = 0;
    long double py = 0;
    long double pz = 0;     
    long double e  = 0; 
    long double m  = 0;
    long double b  = 0;
    long double p  = 0; 

    particle_template* ptr_ = nullptr; 
}; 

#endif
