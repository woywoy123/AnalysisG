#ifndef MULTISOL_CONUIC_CONSTANTS_H
#define MULTISOL_CONUIC_CONSTANTS_H

#include <conuic/atomics.h>
#include <common/matrix.h>
#include <conuic/data.h>

branches_t* build_branches(kinematics_t* j, kinematics_t* l, int sign);
delta_t*    build_deltas(branches_t* plus, branches_t* minus); 
geometry_t build_ruling(branches_t* pls, branches_t* msn);
void        build_tilde(branches_t* br, kinematics_t* knl);

special_t*  build_special(branches_t* plus, branches_t* minus, delta_t* dt, kinematics_t* kln, conuic* db); 


#endif
