#ifndef H_NUNU_CXX
#define H_NUNU_CXX
#include <templates/particle_template.h>

class mtx; 
class nusol; 

struct nunu_t {
    nunu_t();
    ~nunu_t();

    mtx* nu1 = nullptr; 
    mtx* nu2 = nullptr; 
    mtx* agl = nullptr; 
    nusol* nux1 = nullptr; 
    nusol* nux2 = nullptr; 
}; 

class nunu_solver 
{
    public:
        ~nunu_solver();
        nunu_solver(std::vector<particle_template*>* targets, double met, double phi); 
        void nunu_make(particle_template** nu1, particle_template** nu2, double limit);
        void prepare(double mt, double mw); 
        void solve(); 

    private: 
        std::map<int, std::tuple<nusol*, nusol*>> pairings; 
        std::map<double, nunu_t> solvs = {}; 

        std::vector<particle_template*> bquarks = {}; 
        std::vector<particle_template*> leptons = {};
        std::vector<nusol*> engines = {}; 

        void flush(); 
        void make_neutrinos(mtx* v, mtx* v_); 
        int generate(nusol* nu1, nusol* nu2); 
        int intersection(mtx** v, mtx** v_);
        int angle_cross( mtx** v, mtx** v_);

        nusol* p_nu1 = nullptr;
        nusol* p_nu2 = nullptr; 
        
        mtx* m_nu1 = nullptr;
        mtx* m_nu2 = nullptr;
        mtx* m_agl = nullptr; 

        int m_lx = 0;
        int m_bst = -1; 

        double _metx = 0;
        double _mety = 0; 
        double _metz = 0;

        int n_lp = 0;
        int n_bs = 0; 
}; 

#endif

