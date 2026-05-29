#ifndef MULTISOL_CONUIC_ATOMICS_H
#define MULTISOL_CONUIC_ATOMICS_H
#include <conuic/base.h>

template <typename g>
void flush(g** data){
    if (!*data){return;}
    delete *data; *data = nullptr; 
};

long double convert(double v); 
long double nsqrt(long double); 
long double psqrt(long double); 

long double tn_sin(long double v); // sin from tan
long double cs_sin(long double v); // sin from cos
                                  
long double sn_cos(long double v); // cos from sin
long double tn_cos(long double v); // cos from tan
                                   
long double cs_tan(long double v); // tan from cos
long double sn_tan(long double v); // tan from sin

long double pw(long double v, int mul = 2); 
long double dots(kinematics_t* v1, kinematics_t* v2); 
long double costheta(kinematics_t* bq, kinematics_t* lp); 

#endif
