#ifndef GRAPH_FEATURES_BSM_4TOPS_H
#define GRAPH_FEATURES_BSM_4TOPS_H
#include <templates/event_template.h>
class bsm_4tops; 

// ------------------ Truth Graph Features --------------------- //
void num_tops(int* o, bsm_4tops* event);
void num_lepton(int* o, bsm_4tops* event);
void num_neutrino(int* o, bsm_4tops* event);
void signal_event(bool* o, bsm_4tops* event); 

// --------------------- Graph Observables --------------------- //
void missing_et(double* o, bsm_4tops* event);   
void missing_phi(double* o, bsm_4tops* event);  

void num_leps(double* o, bsm_4tops* event); 
void num_jets(double* o, bsm_4tops* event); 
void num_quark(double* o, bsm_4tops* event); 
void num_truthjets(double* o, bsm_4tops* event); 
void num_children_leps(double* o, bsm_4tops* event); 

void event_number(long* o, bsm_4tops* event); 
void event_weight(double* o, bsm_4tops* event);

#endif
