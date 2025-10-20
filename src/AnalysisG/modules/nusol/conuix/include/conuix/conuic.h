#ifndef H_CONUIC
#define H_CONUIC
#include <conuix/struct.h>
#include <structs/property.h>
#include <complex>
#include <math.h>

class particle_template; 








class conuic {
    
    public:
        conuic(particle_template* jet, particle_template* lep); 
        ~conuic(); 

        cproperty<long double, conuic> t; 
        cproperty<long double, conuic> z; 
        cproperty<long double, conuic> l; 

        void solve();  

        // characteristic polynomial of H_tilde
        long double P(); 
        std::complex<long double> P(std::complex<long double> _l, long double _z, long double _t); 

        long double dPdL(); 
        long double dPdZ(); 
        long double dPdtau(); 
    
        // hyperbolic factor --- see source code.
        long double gxx();
        long double gxx(long double _t);

        long double gtx(); 
        long double gtx(long double _t); 

        long double dtx(); 
        long double dtx(long double _t); 
        long double kappa(long double _t, bool use_u = false); 
        long double Mobius(long double _t, bool use_u = false, bool check_u = false); 

        std::complex<long double> lambda_dPdZ(
                long double _z, long double _t, 
                std::complex<long double>* lp, std::complex<long double>* Pp, 
                std::complex<long double>* lm, std::complex<long double>* Pm
       ); 

        std::complex<long double> lambda_dPdL(
                long double _z, long double _t,
                std::complex<long double>* lp, std::complex<long double>* Pp, 
                std::complex<long double>* lm, std::complex<long double>* Pm
        ); 

        void lambda_dPdtau(
                long double _z, long double _t,
                std::complex<long double>* lt, std::complex<long double>* Pt
        ); 


        // P = 0 and dP/dtau = 0.
        // M(tau)^2 = - cosh(tau) / ( cos^2(psi) beta_mu kappa(tau) )
        void lambda_root_dPdtau(
                long double _z, long double _t,
                std::complex<long double>* lt, std::complex<long double>* Pt,
                bool use_numerical = false
        ); 



    private:
        atomics_t* cache = nullptr;
        particle_template* _jet = nullptr; 
        particle_template* _lep = nullptr; 

        void static set_tau(long double* v, conuic* c); 
        void static set_scl(long double* v, conuic* c); 
        void static set_lmb(long double* v, conuic* c); 

        void static get_tau(long double* v, conuic* c); 
        void static get_scl(long double* v, conuic* c); 
        void static get_lmb(long double* v, conuic* c); 

        long double tau  = 0; // simple hyperbolic parameter
        long double stau = 0; // pre-compute sinh
        long double ctau = 0; // pre-compute cosh
        long double ttau = 0; // pre-compute tanh

        long double scale = 0;
        long double lamb  = 0; 

}; 


#endif

