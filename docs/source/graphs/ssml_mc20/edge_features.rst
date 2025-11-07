edge_features.h
===============

**File Path**: ``src/AnalysisG/graphs/ssml_mc20/include/ssml_mc20/edge_features.h``

**File Type**: C++ Header

**Lines**: 75

Description
-----------

--------------------- Edge Truth --------------------- //
void static m_res_edge(int* o, jet* t){*o *= t -> from_res;}
void static m_res_edge(int* o, electron* t){*o *= t -> from_res;}
void static m_res_edge(int* o, muon* t){*o *= t -> from_res;}
void static m_res_edge(int* o, particle_template* t){

