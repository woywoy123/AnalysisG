#ifndef H_ANGULAR_CONUIC
#define H_ANGULAR_CONUIC
#include <common/matrix.h>

class particle_template; 
struct kinematic_c; 

struct angular_t {
    angular_t(); 
    angular_t(long double phi);
    angular_t(long double phi, bool is_cos, bool is_sin, bool is_tan); 

    long double cos = 0;
    long double sin = 0;
    long double tan = 0; 
    long double phi = 0;

    matrix_t Rz(); 
    matrix_t Ry(); 
    matrix_t Rx();
}; 

struct hyper_t {
    hyper_t(); 
    hyper_t(long double tau); 
    long double cosh = 0;
    long double sinh = 0;
    long double tanh = 0; 
    long double tau  = 0; 
    matrix_t mat; 
}; 

struct shift_t {
    shift_t(long double sx_, long double sy_); 
    shift_t(long double sx_, long double sy_, long double sz_); 

    matrix_t to_mat(int dim = -1); 
    void print(); 

    long double sx = 0;
    long double sy = 0;
    long double sz = 0; 
    int dim = -1; 
}; 


long double angles(particle_template* jet, particle_template* lep, kinematic_c* data); 

#endif

