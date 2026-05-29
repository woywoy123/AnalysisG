#ifndef MULTISOL_CONUIC_CONSTANTS_H
#define MULTISOL_CONUIC_CONSTANTS_H
#include <conuic/base.h>

enum class angle_t {undef, cos, sin, tan }; 

struct shared_t : public debug_t {
    shared_t(kinematics_t* bq, kinematics_t* lp); 
    ~shared_t(); 

    long double cos = 0; 
    long double sin = 0;
    long double tan = 0; 
    long double theta = 0;
 
    // kinematics 
    long double r = 0; 
    long double m_mu = 0; 
    long double e_mu = 0; 
    long double b_mu = 0; 
    long double p_mu = 0; 

    long double m_bq = 0; 
    long double e_bq = 0; 
    long double b_bq = 0; 
    long double p_bq = 0; 
}; 


struct base_t : public debug_t {
    base_t(shared_t* shr, long double sign); 
    ~base_t(); 

    long double b_mu = 0; 
    long double e_mu = 0; 

    long double w = 0; 
    long double O = 0; 
    
    long double A = 0; 
    long double B = 0; 
    long double C = 0;
    long double D = 0;
    long double E = 0; 

}; 

struct pk1l_t : public debug_t {
    pk1l_t(base_t* pl, base_t* ms); 
    ~pk1l_t(); 

    long double GP = 0;
    long double GM = 0; 

    long double dp = 0;
    long double dm = 0;  
    long double dpm = 0; 

    long double gmu = 0; 
    long double eta = 0; 
    long double kap = 0; 
    long double kam = 0; 

    long double L0pp = 0; 
    long double L0pm = 0; 
    long double L0mp = 0; 
    long double L0mm = 0; 

    long double lx(long double _sx, long double _sy);
    long double ly(long double _sx, long double _sy);

    long double sx(long double _lx, long double _ly);
    long double sy(long double _lx, long double _ly);

    long double Lx(long double _sx, long double _sy);
    long double Ly(long double _sx, long double _sy);

    long double Sx(long double _lx, long double _ly);
    long double Sy(long double _lx, long double _ly);







}; 




#endif 
