#ifndef H_CONUIX
#define H_CONUIX

#include <reconstruction/nusol.h>

struct nusol_t; 
class conuic; 

class conuix {

    public: 
        conuix(nusol_t* params); 
        ~conuix(); 
        std::vector<particle_template*> nunu_make(); 
        std::vector<conuic*>* cnx = nullptr; 
        nusol_t* params = nullptr;
        std::string prefix = "";
}; 


#endif
