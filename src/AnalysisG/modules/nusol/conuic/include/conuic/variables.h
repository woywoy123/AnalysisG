#ifndef H_VARIABLES_CONUIC
#define H_VARIABLES_CONUIC
#include <templates/particle_template.h>
#include <common/matrix.h>

#include <conuic/variables.h>
#include <conuic/constants.h>
#include <conuic/branches.h>
#include <conuic/angular.h>
#include <conuic/atomics.h>
#include <conuic/pencil.h>
#include <conuic/factor.h>
#include <conuic/base.h>


struct kinematic_c {
    ~kinematic_c(); 
    kinematic_c(particle_template* jet, particle_template* lep);  
    void debug(); 

   // ------- angles ------- //
    // - Theta
    angular_t theta; 
    matrix_t* rot = nullptr;  

    Z2_t Zpp;
    Z2_t Zpm;
    Z2_t Zmp;
    Z2_t Zmm; 
    G2_t G2; 
 
    // ------- constants  ----------//
    branches_t O; 
    branches_t w; 


    // lepton kinematics
    long double p_mu = 0; 
    long double m_mu = 0; 
    long double b_mu = 0; 
    long double e_mu = 0; 

    // jet kinematics
    long double e_b  = 0;
    long double p_b  = 0; 
    long double m_b  = 0; 
    long double b_b  = 0; 

    particle_template* jet = nullptr;
    particle_template* lep = nullptr; 

}; 


#endif
