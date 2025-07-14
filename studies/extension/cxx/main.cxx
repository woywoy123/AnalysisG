#include "nunu.h"
#include "lm.h"

int main(){
    particle* b1  = new particle(-19.766428,  -40.022249,   69.855886,  83.191328);
    particle* b2  = new particle(107.795878, -185.326183,  -67.989162, 225.794953);
    particle* mu1 = new particle(-14.306453,  -47.019613,    3.816470,  49.295996);
    particle* mu2 = new particle(  4.941336, -104.097506, -103.640669, 146.976547);
    double metx = 106.435841; 
    double mety = -141.293331;

    double mt1 = 172.68;
    double mt2 = 172.68; 
    double mw1 = 80.385; 
    double mw2 = 80.385; 
    nunu* nx = new nunu(b1, b2, mu1, mu2, mt1, mt2, mw1, mw2, false); 
    double** params = matrix(1, 4); 
    params[0][0] = mt1; params[0][2] = mt2;
    params[0][1] = mw1; params[0][3] = mw2;
    nx -> metx = metx; 
    nx -> mety = mety; 

    //nx -> generate(metx, mety, 0);

    LevenbergMarquardt* lm = new LevenbergMarquardt(nx, params, 0.001, 1e3, 1e-3, 100);
    lm -> optimize(); 


    delete nx; 
    return 0; 
}
