#ifndef H_NUSOL
#define H_NUSOL

#include <tools/tools.h>
#include <notification/notification.h>
#include <templates/particle_template.h>

class nusol; 
class ellipse; 
class conics; 

enum class nusol_enum {
    ellipse,  // https://arxiv.org/pdf/1305.1878
    conics,   // generalized n-neutrino approach
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
        double limit = 10000; 

        nusol_enum mode = nusol_enum::undefined; 
        std::vector<particle_template*>* targets = nullptr;  

    private:
        friend nusol; 
        friend ellipse; 
        friend conics;

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
        void solve(); 
        ~nusol(); 

    private:
        nusol_t* params = nullptr; 
        ellipse* D_nunu = nullptr; 
        conics*  M_nunu = nullptr; 
}; 

#endif
