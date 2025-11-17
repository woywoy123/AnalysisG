#include <reconstruction/nusol.h>
#include <ellipse/ellipse.h>
#include <conuix/conuix.h>

nusol::nusol(nusol_t* parameters){
    this -> params = parameters; 
    this -> prefix = "NuSol"; 

    this -> params -> met_x = std::cos(this -> params -> phi) * this -> params -> met;  
    this -> params -> met_y = std::sin(this -> params -> phi) * this -> params -> met;
}

nusol::~nusol(){
    if (this -> D_nunu){delete this -> D_nunu;}
    if (this -> M_nunu){delete this -> M_nunu;}
}

std::vector<particle_template*> nusol::solve(){
//    switch (this -> params -> mode){
//        case nusol_enum::ellipse: this -> D_nunu = new ellipse(this -> params); break;
//        case nusol_enum::conuix:  this -> M_nunu = new conuix( this -> params); break; 
//        default: break;  
//    }   
    this -> M_nunu = new conuix( this -> params);    
    this -> D_nunu = new ellipse(this -> params);
    this -> D_nunu -> prepare(this -> params -> mt, this -> params -> mw); 
    this -> D_nunu -> solve(); 
    return this -> D_nunu -> nunu_make(); 
}

