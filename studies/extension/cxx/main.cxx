#include <templates/nunu.h>

int main(){

    wrapper* b1  = new wrapper(-19.766428,  -40.022249,   69.855886,  83.191328);
    wrapper* b2  = new wrapper(107.795878, -185.326183,  -67.989162, 225.794953);
    wrapper* mu1 = new wrapper(-14.306453,  -47.019613,    3.816470,  49.295996);
    wrapper* mu2 = new wrapper(  4.941336, -104.097506, -103.640669, 146.976547);

    mu1 -> is_lep = true; b1 -> is_b = true;
    mu2 -> is_lep = true; b2 -> is_b = true;

    std::vector<wrapper*> targets = {b1, b2, mu1, mu2}; 
    double met = 106.435841; 
    double phi = -141.293331;
    
    nunu_solver* sol = new nunu_solver(&targets, met, phi); 
    sol -> prepare(172.68, 80.384);
    sol -> solve(); 
    delete sol;
    return 0; 
}; 

