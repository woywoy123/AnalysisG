#ifndef H_ELLIPSE
#define H_ELLIPSE

#include <templates/particle_template.h>
#include <ellipse/solvers.h>
#include <ellipse/nusol.h>
#include <ellipse/mtx.h>

// forward declare the main interface struct.
struct nusol_t; 

struct nunu_t {
    nunu_t();
    ~nunu_t();

    mtx* nu1 = nullptr; 
    mtx* nu2 = nullptr; 
    mtx* agl = nullptr; 
    nuelx* nux1 = nullptr; 
    nuelx* nux2 = nullptr; 
}; 


class ellipse {

    public:
        ellipse(nusol_t* parameters); 
        void prepare(double mt, double mw); 
        std::vector<particle_template*> nunu_make();
        void solve(); 
        ~ellipse(); 

        template <typename g>
        void clear(g** val){
            if (!*val){return;}
            delete *val; 
            *val = nullptr; 
        }

    private:
        nusol_t* params = nullptr; 

        std::map<int, std::tuple<nuelx*, nuelx*>> pairings; 
        std::map<double, nunu_t> solvs = {}; 

        std::vector<particle_template*> bquarks = {}; 
        std::vector<particle_template*> leptons = {};
        std::vector<nuelx*> engines = {}; 

        void flush(); 
        void make_neutrinos(mtx* v, mtx* v_); 
        int generate(nuelx* nu1, nuelx* nu2); 
        int intersection(mtx** v, mtx** v_);
        int angle_cross( mtx** v, mtx** v_);

        nuelx* p_nu1 = nullptr;
        nuelx* p_nu2 = nullptr; 
        
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
