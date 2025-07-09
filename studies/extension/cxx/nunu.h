#ifndef H_NUNU
#define H_NUNU
#include <iostream>
#include <iomanip>
#include <cmath>

class particle; 
double costheta(particle* p1, particle* p2); 
double sintheta(particle* p1, particle* p2);
void clear(double** mx, int row, int col); 
double** matrix(int row, int col); 
void print(double** mx); 
double** dot(double** v1, double** v2, int r1, int c1, int r2, int c2); 
double** T(double** v1, int r, int c); 

class particle {
    public:
        particle(double px, double py, double pz, double e); 
        ~particle();
        double p(); 
        double p2(); 
        double m2(); 
        double beta();
        double phi(); 
        double theta(); 

        double px = 0; 
        double py = 0;
        double pz = 0;
        double e  = 0; 
}; 


class nusol {
    public:
        nusol(particle* b, particle* l, double mW, double mT); 
        double Sx(); 
        double Sy(); 
        double w();
        double om2(); 
        double x1();
        double y1(); 
        double Z2(); 
        void Z2_coeff(double* A, double* B, double* C); 
        double** H_tilde();
        double** R_T();
        double** H(); 

        ~nusol();  

    private:
        particle* b = nullptr;
        particle* l = nullptr; 
        double mw = 0;
        double mt = 0; 
        double _s = 0; 
        double _c = 0; 
        double** h_tilde = nullptr;
        double** r_t = nullptr;  
        double** h = nullptr; 

}; 






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

#endif


int main(){

    particle* b1  = new particle(-19.766428,  -40.022249,   69.855886,  83.191328);
    particle* b2  = new particle(107.795878, -185.326183,  -67.989162, 225.794953);
    particle* mu1 = new particle(-14.306453,  -47.019613,    3.816470,  49.295996);
    particle* mu2 = new particle(  4.941336, -104.097506, -103.640669, 146.976547);

    nunu* nx = new nunu(b1, b2, mu1, mu2); 
    nx -> generate();

//-2.952913482255793 -779.6828478607471 -0.8755574794686254


//[[385.56426537   0.         387.11260369]
// [195.99708945   0.         128.92644264]
// [  0.         195.99883329   0.        ]]

//[[-0.29108966 -0.95669578  0.        ]
// [ 0.95669578 -0.29108966  0.        ]
// [ 0.          0.          1.        ]]
//
//[[ 0.99699859  0.          0.07741965]
// [ 0.          1.          0.        ]
// [-0.07741965  0.          0.99699859]]
//[44.04291941 -7.26039533 69.855886  ]

    delete nx; 
    return 0; 
}

