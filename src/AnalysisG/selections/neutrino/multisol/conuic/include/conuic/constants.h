#ifndef MULTISOL_CONUIC_CONSTANTS_H
#define MULTISOL_CONUIC_CONSTANTS_H

#include <conuic/atomics.h>
#include <common/matrix.h>
#include <conuic/data.h>

branches_t* build_branches(kinematics_t* j, kinematics_t* l, int sign);
delta_t*    build_deltas(branches_t* plus, branches_t* minus); 
void        build_tilde(branches_t* br, kinematics_t* knl);

cline_t*    build_clines(branches_t* br, delta_t* dt, double sign); 

#endif
