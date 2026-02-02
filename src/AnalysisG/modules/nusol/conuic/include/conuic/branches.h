#ifndef H_BRANCHES_CONUIC
#define H_BRANCHES_CONUIC

struct kinematic_c;
class particle_template; 

long double angles(particle_template* jet, particle_template* lep, kinematic_c* data); 
long double omega(long double sign, kinematic_c* data); 
long double Omega(int sign, kinematic_c* data); 

long double pencil(int sign, kinematic_c* data); 


//long double delta(int sign, kinematic_c* data); 

long double  signs(int sign, long double  v1, long double  v2);
long double* signs(int sign, long double* v1, long double* v2);


#endif


