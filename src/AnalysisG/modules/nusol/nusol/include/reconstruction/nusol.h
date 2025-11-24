#ifndef H_NUSOL
#define H_NUSOL

#include <tools/tools.h>
#include <notification/notification.h>
#include <templates/particle_template.h>

class nusol; 
class ellipse; 
class conuix; 
class nuelx; 
class mtx; 

enum class nusol_enum {
    ellipse,  // https://arxiv.org/pdf/1305.1878
    conuix,   // generalized n-neutrino approach
    undefined
}; 

// event parameters
struct nusol_t {
    public: 
        // ----- set these ----- //
        double met = 0; 
        double phi = 0; 

        // ----- for the ellipse method ---- //
        double mt = 172.68 * 1000;
        double mw = 80.385 * 1000; 
        double violation = 0.00001; 
        double limit = 0.1; 
        int iterations = 10; 

        nusol_enum mode = nusol_enum::undefined; 
        std::vector<particle_template*>* targets = nullptr;  
        std::vector< std::pair<particle_template*, particle_template*> >* phys_pairs = nullptr; 

    private:
        friend ellipse; 
        friend conuix;
        friend nusol; 

        double met_x = 0; 
        double met_y = 0; 
        double met_z = 0; 
}; 


class nusol: 
    notification, 
    tools
{
    public:
        nusol(nusol_t* parameters); 
        std::vector<particle_template*> solve(); 
        ~nusol(); 

    private:
        class neutrino : public particle_template {
            public: 
                using particle_template::particle_template; 
                virtual ~neutrino(); 
                neutrino(nuelx* nu, mtx* mx); 
                void merge(neutrino* nx); 
                particle_template* release(); 
                double score = 0; 
        }; 


        nusol_t* params = nullptr; 
        ellipse* D_nunu = nullptr; 
        conuix*  M_nunu = nullptr; 
}; 

#endif
