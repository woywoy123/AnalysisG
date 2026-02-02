#include <templates/particle_template.h>
#include <conuic/variables.h>
#include <conuic/branches.h>
#include <common/matrix.h>
#include <math.h>

kinematic_c::kinematic_c(
    particle_template* jet, particle_template* lep
){
    this -> cth = angles(jet, lep, this); 
    this -> sth = std::sqrt(1 - cth * cth); 
    this -> tth = sth / cth; 
    
    this -> wp = omega(+1, this); this -> wm = omega(-1, this);
    this -> op = Omega(+1, this); this -> om = Omega(-1, this); 
   
}








kinematic_c::~kinematic_c(){
    flush(&this -> rot); 
}


