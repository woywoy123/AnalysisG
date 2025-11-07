edge_features.h
===============

**File Path**: ``src/AnalysisG/graphs/bsm_4tops/include/bsm_4tops/edge_features.h``

**File Type**: C++ Header

**Lines**: 12

Description
-----------

--------------------- Edge Truth --------------------- //
void res_edge(int* o, std::tuple<particle_template*, particle_template*>* pij);
void top_edge(int* o, std::tuple<particle_template*, particle_template*>* pij);
void det_top_edge(int* o, std::tuple<particle_template*, particle_template*>* pij);
void det_res_edge(int* o, std::tuple<particle_template*, particle_template*>* pij);

Dependencies
------------

**C++ Includes**:

- ``templates/event_template.h``

