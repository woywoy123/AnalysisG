#include <multisol/conuix.h>
#include <multisol/solvers.h>

conuic::~conuic(){}
conuic::conuic(particle_template* jt, particle_template* lep) : cache(jt, lep){
    this -> state = this -> init(); 
}

void conuic::intersection(
    conuic* nux,    long double metx, long double mety, 
    long double t1, long double z1  , long double t2, long double z2
){
    z1 += (!z1) ? 1 : 0; z2 += (!z2) ? 1 : 0;  
    nux -> hyper(t1); this -> hyper(t2);
    matrix_t N1 =  nux -> N( nux -> _ct,  nux -> _st, 1.0, false);
    matrix_t N2 = this -> N(this -> _ct, this -> _st, 1.0, false);
   
    matrix_t T = S0(metx / z2, mety / z2);
    T.at(0, 0) = -z1 / z2;
    T.at(1, 1) = -z1 / z2;
    
    matrix_t n_ = T.T().dot(N2).dot(T);
    matrix_t* nu = intersection_ellipses(&N1, &n_);
    std::cout << nu << " " << z1 << " " << z2 << " " << metx << " " << mety << std::endl;
    if (!nu){return;}

    for (int x(0); x < nu -> r; ++x){
        matrix_t nu1 = nu -> at(x).T();  
        matrix_t nu2 = T.dot(nu1); 

        nu1 =  nux -> Nu(nu1, z1, false); 
        nu2 = this -> Nu(nu2, z2, false); 

        matrix_t Wnux =  nux -> make_w(&nu1);
        matrix_t Wnuy = this -> make_w(&nu2);

        matrix_t Tnux =  nux -> make_top(&nu1);
        matrix_t Tnuy = this -> make_top(&nu2);
        (nu1 * (0.001L)).cat(nu2 * (0.001L)).print(); 

        long double metx_ = nu1.at(0, 0) + nu2.at(0, 0);
        long double mety_ = nu1.at(0, 1) + nu2.at(0, 1);

        std::cout << metx_ << " " << metx << " " << mety_ << " " << mety << std::endl;

        std::cout << "-------" << std::endl; 
        std::cout << "+ < "; 
        std::cout <<  nux -> get_mass(&Wnux).real() * 0.001L << " ";
        std::cout << this -> get_mass(&Wnuy).real() * 0.001L << " "; 
        std::cout <<  nux -> get_mass(&Tnux).real() * 0.001L << " "; 
        std::cout << this -> get_mass(&Tnuy).real() * 0.001L << " "; 
        std::cout << std::endl; 
    }
    delete nu; 
}


