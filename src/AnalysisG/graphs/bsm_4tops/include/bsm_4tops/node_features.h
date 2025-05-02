#ifndef NODES_FEATURES_BSM_4TOPS_H
#define NODES_FEATURES_BSM_4TOPS_H
#include <templates/event_template.h>

// --------------------- Node Truth --------------------- //
void res_node(int* o, particle_template* p); 
void top_node(int* o, particle_template* p); 

// --------------------- Node Observables --------------------- //
void pt(double* o, particle_template* p); 
void eta(double* o, particle_template* p);  
void phi(double* o, particle_template* p);  
void energy(double* o, particle_template* p);
void charge(double* o, particle_template* p);

void is_lepton(int* o, particle_template* p); 
void is_bquark(int* o, particle_template* p);
void is_neutrino(int* o, particle_template* p);

#endif
