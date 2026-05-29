#ifndef MULTISOL_CONUIC_H
#define MULTISOL_CONUIC_H
#include <templates/particle_template.h>
#include <conuic/constants.h>
#include <common/matrix.h>
#include <conuic/base.h>

class conuic {
    public:
        conuic(particle_template* jet, particle_template* lepton);
        void proof(particle_template* nux); 

        long double mW2(long double sx, long double m_nu); 
        long double mT2(long double sx, long double sy, long double m_nu); 

        long double Sx(long double mW, long double m_nu); 
        long double Sy(long double mT, long double mW, long double m_nu); 

        long double Z2(
                long double sx, long double sy, 
                long double m_nu, int sign
        );

        long double Z2lxly(
                long double   lx, long double ly, 
                long double m_nu, long double sign
        ); 

        long double g2(long double sx, long double sy); 
        long double G2(long double sx, long double sy); 
        ~conuic();

    private: 
        base_t* branching(int s); 

        kinematics_t* jet_  = nullptr; 
        kinematics_t* lep_  = nullptr;  
        matrix_t*     RT_   = nullptr; 

        shared_t* shr = nullptr; 
        base_t* plus  = nullptr; 
        base_t* minus = nullptr; 
        pk1l_t* pl1   = nullptr; 

};

#endif
