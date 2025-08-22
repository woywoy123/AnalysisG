#ifndef HXX_NUSOL
#define HXX_NUSOL

class mtx; 
class particle_template;

struct wrapper 
{
    wrapper(particle_template* p_);
    double p  = 0; 
    double m  = 0;
    double b  = 0; 
    double p2 = 0; 
    double m2 = 0; 
    double b2 = 0; 
    double phi = 0;
    double theta = 0;
    double px = 0;
    double py = 0;
    double pz = 0;
    double e  = 0;
    particle_template* lnk = nullptr; 
}; 


class nuelx 
{
    public:
        nuelx(); 
        nuelx(particle_template* b, particle_template* l, double mW, double mT); 
        ~nuelx(); 

        double Sx(); 
        double dSx_dmW();

        double Sy(); 
        double dSy_dmW(); 
        double dSy_dmT(); 

        double w();
        double w2(); 
        double om2(); 

        double Z(); 
        double Z2(); 
        double dZ_dmT(); 
        double dZ_dmW(); 

        double x0(); 
        double x1();
        double dx1_dmW(); 
        double dx1_dmT(); 

        double  y1(); 
        double dy1_dmW(); 
        double dy1_dmT(); 

        void r_mT(double* mt1_, double* mt2_); 
        void r_mW(double* mw1_, double* mw2_); 
        void Z_mW(double* mw1_, double* mw2_); 
        void Z_mT(double* mt1_, double* mt2_, double mw_); 

        void update(double mt, double mw);
        void flush(); 

        mtx* N(); 
        mtx* H(); 
        mtx* H_perp(); 
        mtx* H_tilde();
        mtx* dH_dmW();
        mtx* dH_dmT(); 
        mtx* K(); 
        mtx* R_T();

        void Z2(double* A, double* B, double* C); 
        void misc(); 

        wrapper* b = nullptr;
        wrapper* l = nullptr; 
    private:

        double mw = 0;
        double mt = 0; 
        double mw2 = 0; 
        double mt2 = 0; 

        double _s = 0; 
        double _c = 0; 

        mtx* h_tilde = nullptr; 
        mtx* h       = nullptr; 
        mtx* r_t     = nullptr; 
        mtx* dw_H    = nullptr; 
        mtx* dt_H    = nullptr; 
        mtx* h_perp  = nullptr; 
        mtx* n_matrx = nullptr; 
        mtx* k       = nullptr; 

}; 

#endif
