#ifndef H_VARIABLES_CONUIC
#define H_VARIABLES_CONUIC

#include <templates/particle_template.h>
#include <common/matrix.h>

template <typename g>
void flush(g** data){
    if (!*data){return;}
    delete *data; *data = nullptr; 
}; 


struct kinematic_c {
    kinematic_c(particle_template* jet, particle_template* lep);  
    ~kinematic_c(); 

    // lepton kinematics
    long double p_mu = 0; 
    long double m_mu = 0; 
    long double b_mu = 0; 

    // jet kinematics
    long double p_b  = 0; 
    long double m_b  = 0; 
    long double b_b  = 0; 

    // ------- angles ------- //
    // - Theta
    long double cth  = 0; 
    long double sth  = 0; 
    long double tth  = 0;  

    // ------- constants  ----------//
    long double op = 0; 
    long double om = 0;

    long double wp = 0; 
    long double wm = 0; 

    // ------ Z2 branches -------- //
    long double z2p_a = 0; 
    long double z2p_b = 0; 
    long double z2p_c = 0; 
    long double z2p_d = 0; 
    long double z2p_e = 0; 

    long double z2m_a = 0; 
    long double z2m_b = 0; 
    long double z2m_c = 0; 
    long double z2m_d = 0; 
    long double z2m_e = 0; 

    matrix_t* rot = nullptr;  
}; 


#endif
