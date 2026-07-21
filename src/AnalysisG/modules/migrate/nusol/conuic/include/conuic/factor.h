#ifndef H_FACTOR_CONUIC
#define H_FACTOR_CONUIC

#include <conuic/branches.h>
#include <conuic/angular.h>
#include <common/matrix.h>
struct kinematic_c; 

struct G2_t {
    G2_t();
    G2_t(kinematic_c* data);

    shift_t dG2(long double tau);
    long double dG2(long double sx, long double sy);
    long double tau(long double sx, long double sy); 

    // ------- Delta values ------- //
    branches_t delta; // delta roots.
    branches_t kappa; // delta as angles 

    // 0.5 * (psi^+ + psi^-)
    // 0.5 * (psi^+ - psi^-)
    branches_t psi; 

    // ------ Eigenvalues --------- //
    branches_t G, lambda, O, w; 
    matrix_t eigM, MK; 

    long double b_mu  = 0;
    long double m_mu  = 0;
    long double p_mu  = 0;

    long double b2_mu = 0; 
    long double m2_mu = 0; 

}; 




#endif
