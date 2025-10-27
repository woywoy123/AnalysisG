#ifndef H_CONUIX_STRUCT_H
#define H_CONUIX_STRUCT_H
#include <conuix/matrix.h>
class particle_template; 

namespace Conuix {
    struct kinematic_t; 
    struct rotation_t; 
    struct base_t; 
    struct thetapsi_t; 
    struct pencil_t; 
    struct Sx_t; 
    struct Sy_t; 
    struct H_matrix_t; 

    void get_kinematic(particle_template* ptr, kinematic_t* kin); 

    void get_psi_theta_mapping(base_t* bs, thetapsi_t* msp); 
    void get_rotation(particle_template* jet, particle_template* lep, rotation_t* rot);

    void get_base(kinematic_t* jet, kinematic_t* lep, base_t* bs); 
    void get_pencil(kinematic_t* lep, kinematic_t* nu, base_t* base, pencil_t* pen); 

    void get_sx(base_t* bs, Sx_t* sx); 
    void get_sy(base_t* bs, Sy_t* sy); 
 
    void get_hmatrix(base_t* base, rotation_t* rot, H_matrix_t* H); 
}


#endif



