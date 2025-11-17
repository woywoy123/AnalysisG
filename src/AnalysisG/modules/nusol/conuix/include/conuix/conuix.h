#ifndef H_CONUIX
#define H_CONUIX

#include <reconstruction/nusol.h>

struct nusol_t; 
class conuic; 

class conuix {

    public: 
        conuix(nusol_t* params); 
        ~conuix(); 
//        void solve(); 
    
    private:
        std::string prefix = "";
        nusol_t* params = nullptr;
        std::vector<conuic*>* cnx = nullptr; 
}; 


#endif
