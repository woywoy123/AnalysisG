#ifndef MULTISOL_CONUIC_H
#define MULTISOL_CONUIC_H

#include <conuic/data.h>
#include <conuic/atomics.h>
#include <templates/particle_template.h>

class conuic {
    public:

        conuic(particle_template* jet, particle_template* lepton);
        long double Z2(long double sx,  long double sy   , long double m_nu, int sign); 
        long double Sx(long double tau, long double kappa, long double m_nu, int sign, int eps); 
        long double Sy(long double tau, long double kappa, long double m_nu, int sign, int eps); 

        long double Z(long double tau, long double kappa, long double m_nu, int sign); 
        long double x1(long double tau, long double kappa, long double m_nu, int sign, int eps); 
        long double y1(long double tau, long double kappa, long double m_nu, int sign, int eps); 

        points_t S(long double tau, long double kappa, long double m_nu, int sign, int eps);
        matrix_t H_tilde(long double tau, long double kappa, long double m_nu, int sign, int eps);

        long double line(long double sx, long double sy, int sign); 
        long double dG2(long double sx, long double sy); 

        branches_t* brn(int sign); 
        ~conuic();

    private: 
         
        kinematics_t* jet_  = nullptr; 
        kinematics_t* lep_  = nullptr;  
        branches_t*   plus  = nullptr;
        branches_t*   minus = nullptr; 
        delta_t*      delG  = nullptr; 
        special_t*    splx  = nullptr; 
        matrix_t*     RT    = nullptr; 

};

#endif
