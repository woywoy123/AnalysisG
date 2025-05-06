#ifndef EDGE_FEATURES_BSM_4TOPS_H
#define EDGE_FEATURES_BSM_4TOPS_H
#include <templates/event_template.h>

// --------------------- Edge Truth --------------------- //
void res_edge(int* o, std::tuple<particle_template*, particle_template*>* pij);
void top_edge(int* o, std::tuple<particle_template*, particle_template*>* pij);
void det_top_edge(int* o, std::tuple<particle_template*, particle_template*>* pij); 
void det_res_edge(int* o, std::tuple<particle_template*, particle_template*>* pij); 

#endif
