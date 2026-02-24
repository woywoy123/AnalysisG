#ifndef H_CONUIAC
#define H_CONUIAC
#include <reconstruction/nusol.h>
#include <structs/property.h>
#include <complex>
#include <math.h>

struct kinematic_c; 
class particle_template; 

class conuiac {
    
    public:
        conuiac(nusol_t* param); 

        ~conuiac(); 
        std::vector<particle_template*> solve(); 
        
    private:
        std::vector<kinematic_c*> kins = {}; 
        nusol_t* params = nullptr;
}; 


#endif
