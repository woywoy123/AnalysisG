#ifndef MULTISOL_CONUIC_ANGULAR_H
#define MULTISOL_CONUIC_ANGULAR_H

#include <templates/particle_template.h>
#include <conuic/constants.h>
#include <common/matrix.h>

matrix_t* angles(particle_template* jet, particle_template* lep); 

struct angular_t {
    angular_t(long double kappa, angle_t agl = angle_t::undef); 

    long double cos = 0;
    long double sin = 0;
    long double tan = 0;
    long double agl = 0; 

    matrix_t Rx(); 
    matrix_t Ry(); 
    matrix_t Rz(); 
}; 

#endif
