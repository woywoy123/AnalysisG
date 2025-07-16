#ifndef H_NUNU
#define H_NUNU
#include "base.h"

class nunu 
{
    public:
        nunu(
                double b1_px, double b1_py, double b1_pz, double b1_e,
                double l1_px, double l1_py, double l1_pz, double l1_e, 
                double b2_px, double b2_py, double b2_pz, double b2_e,
                double l2_px, double l2_py, double l2_pz, double l2_e, 
                double mt1  , double mt2  , double mw1  , double mw2
        ); 

        nunu(
                particle* b1, particle* l1, particle* b2, particle* l2, 
                double mt1, double mt2, double mw1, double mw2, bool delP = true
        ); 

        int generate(double metx, double mety, double metz); 
        void get_nu(particle** nu1, particle** nu2, int l);
        void get_misc(); 
        void _clear(); 
        void flush(); 

        void update(double** params); 
        double** jacobian(int* ix, int* jx); 
        double** loss(int* ix); 
        ~nunu(); 

        double metx = 0; 
        double mety = 0; 


    private:
        int intersection(mtx** v, mtx** v_, double metx, double mety, double metz); 
        int angle_cross( mtx** v, mtx** v_, double metx, double mety, double metz);  

        void make_neutrinos(double** v, double** v_, double* d_, double* agl); 
        particle** make_particle(double** v, double** d, int lx);

        particle** m_v1 = nullptr; 
        particle** m_v2 = nullptr;

        double** m_nu1_ = nullptr;
        double** m_nu2_ = nullptr; 
        double** m_d1_  = nullptr; 
        double** m_agl_ = nullptr; 
        int m_lx = 0; 

        nusol* nu1 = nullptr;
        nusol* nu2 = nullptr; 


}; 

#endif
