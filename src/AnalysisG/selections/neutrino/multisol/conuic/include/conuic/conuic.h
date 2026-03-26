#ifndef MULTISOL_CONUIC_H
#define MULTISOL_CONUIC_H

#include <conuic/data.h>
#include <conuic/atomics.h>
#include <templates/particle_template.h>

class conuic {
    public:

        conuic(particle_template* jet, particle_template* lepton);

        // sign selects the omega branch
        // eps is the sheet index of the hyperboloids.
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

        std::complex<long double> mW(long double sx, long double m_nu); 
        std::complex<long double> mT(long double sx, long double sy, long double m_nu); 
 
        void solve();  
        


        branches_t* brn(int sign); 
        ~conuic();




    private: 
         
        kinematics_t* jet_  = nullptr; 
        kinematics_t* lep_  = nullptr;  
        branches_t*   plus  = nullptr;
        branches_t*   minus = nullptr; 
        delta_t*      delG  = nullptr; 
        matrix_t*     RT    = nullptr; 

        // these are the singular lines, they should be treated as a 
        // pair, i.e. dt_<Y>_eps_x x needs to be the same (sheet index)
        cline_t*      w_x_dt_x_eps_x = nullptr; // +++ configuration 
        cline_t*      w_x_dt_I_eps_x = nullptr; // +-+ configuration
        
        cline_t*      w_x_dt_x_eps_I = nullptr; // ++- configuration
        cline_t*      w_x_dt_I_eps_I = nullptr; // +-- configuration

        cline_t*      w_I_dt_x_eps_x = nullptr; // -++ configuration
        cline_t*      w_I_dt_I_eps_x = nullptr; // --+ configuration
                                               
        cline_t*      w_I_dt_x_eps_I = nullptr; // -+- configuration
        cline_t*      w_I_dt_I_eps_I = nullptr; // --- configuration


        dline_t*      dt_w_x_eps_x  = nullptr; 
        dline_t*      dt_w_x_eps_I  = nullptr; 
        dline_t*      dt_w_I_eps_x  = nullptr; 
        dline_t*      dt_w_I_eps_I  = nullptr; 


};

#endif
