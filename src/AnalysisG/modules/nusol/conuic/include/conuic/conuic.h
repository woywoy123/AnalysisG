#ifndef H_CONUIAC
#define H_CONUIAC
#include <reconstruction/nusol.h>
#include <structs/property.h>
#include <complex>
#include <math.h>

class particle_template; 

class conuiac {
    
    public:
        conuiac(nusol_t* param); 

        ~conuiac(); 
        std::vector<particle_template*> solve(); 

    private:
        particle_template* _jet = nullptr; 
        particle_template* _lep = nullptr; 
}; 


#endif
