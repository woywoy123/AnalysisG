#ifndef H_MULTISOL_MAIN
#define H_MULTISOL_MAIN

#include "multisol/conuix.h"
#include <vector>

struct stats_t {
    int num_lep = 0; 
    int num_jet = 0; 
    int pas_lep = 0; 
    int pas_jet = 0;

    int num_cmb = 0; 
    int num_apt = 0; 
   
    long double low_mob = -1; 
}; 


class multisol {
    public:
        struct vio_t {
            long double mw   = -1;
            long double mt   = -1; 
            long double tau  = -1;
            long double Z    = -1; 

            // meta data 
            long double l0vio  = -1; 
            long double dpdtl0 = -1; 
            long double pl     = -1; 
            long double dpdt   = -1; 

            long double tau_pole[2]; 
            long double dpdz[2];
            void print(int prc = 12); 
        };

        multisol(params_t* prm); 
        ~multisol(); 

        void prescan(); 
        void mass_sample(conuic* nux, long double mw, long double mt);


        multisol::vio_t test(
            conuic* tr, matrix_t* nux, long double mt, long double mw, 
            long double t , long double z
        ); 

        int  violation_test(conuic* v, matrix_t* sols); 
        stats_t metric; 

    private:

        template <typename g>
        void dSafe(std::vector<g*>* v){
            for (int x(0); x < v -> size(); ++x){
                if (!v -> at(x)){continue;}
                delete v -> at(x); 
                v -> at(x) = nullptr; 
            }
        }

        std::map<std::string, std::map<long double, multisol::vio_t>> test_points; 
        std::vector<conuic*> engines; 
        params_t* param = nullptr; 
}; 


#endif
