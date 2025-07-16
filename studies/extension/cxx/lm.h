#ifndef H_LEVENBERG_MARQUARDT
#define H_LEVENBERG_MARQUARDT
#include "nunu.h"

class LevenbergMarquardt
{
    public:
        LevenbergMarquardt(nunu* nux, double** mx, double l0, double lMx, double lMn, long mxiter); 
        double loss(double** param, int lx);
        double** jacobian(double** param); 

        void optimize(); 
        ~LevenbergMarquardt(); 

    private:
        double step();

        nunu* nux_ = nullptr; 
        double lambda0 = 0.001; 
        double lambdaMx = 1e5; 
        double lambdaMn = 1e-5;
        double lambda_c = 0.001; 
        long mxiter = 0; 


        double** _params = nullptr; 
}; 

#endif
