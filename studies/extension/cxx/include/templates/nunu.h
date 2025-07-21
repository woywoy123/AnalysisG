#ifndef H_NUNU_CXX
#define H_NUNU_CXX

#include <templates/nusol.h>
#include <iostream>
#include <vector>
#include <map>

class mtx; 
class nusol; 

struct nunu_t {
    nunu_t();
    ~nunu_t();
    mtx* nu1 = nullptr; 
    mtx* nu2 = nullptr; 
    mtx* agl = nullptr; 
}; 

class nunu_solver 
{
    public:
        ~nunu_solver();
        nunu_solver(std::vector<wrapper*>* targets, double met, double phi); 
        void prepare(double mt, double mw); 
        void solve(); 
        bool verbose = false; 

    private: 
        std::map<double, std::tuple<nusol*, nusol*>> pairings; 
        std::map<double, nunu_t> solvs = {}; 

        std::vector<wrapper*> bquarks = {}; 
        std::vector<wrapper*> leptons = {};
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


