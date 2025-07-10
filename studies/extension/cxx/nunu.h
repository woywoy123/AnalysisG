#ifndef H_NUNU
#define H_NUNU
#include "base.h"

class nunu 
{
    public:
        nunu(particle* b1, particle* l1, particle* b2, particle* l2); 
        void generate(); 
        ~nunu(); 


    private:
        nusol* nu1 = nullptr;
        nusol* nu2 = nullptr; 

}; 



int main(){

    particle* b1  = new particle(-19.766428,  -40.022249,   69.855886,  83.191328);
    particle* b2  = new particle(107.795878, -185.326183,  -67.989162, 225.794953);
    particle* mu1 = new particle(-14.306453,  -47.019613,    3.816470,  49.295996);
    particle* mu2 = new particle(  4.941336, -104.097506, -103.640669, 146.976547);

    nunu* nx = new nunu(b1, b2, mu1, mu2); 
    nx -> generate();

    delete nx; 
    return 0; 
}

#endif
