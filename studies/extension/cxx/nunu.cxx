#include "nunu.h"


nunu::nunu(particle* b1, particle* b2, particle* l1, particle* l2){
    this -> nu1 = new nusol(b1, l1, 76.765395, 142.703747);
    this -> nu2 = new nusol(b2, l2, 93.568769, 164.548815);
}

void nunu::generate(){
    double** n1 = this -> nu1 -> N(); 
    double** n2 = this -> nu2 -> N(); 
    double** S  = smatx(106.435841, -141.293331, 0); 
    double** ST  = T(S, 3, 3);
    double** STN = dot(ST , n2, 3, 3, 3, 3); 
    double** n_  = dot(STN,  S, 3, 3, 3, 3); 
    clear(ST, 3, 3); clear(STN, 3, 3); 
    intersection_ellipses(n1, n_); 
    abort(); 



}

nunu::~nunu(){
    delete this -> nu1; 
    delete this -> nu2; 
}

