particles.cxx
=============

**File Path**: ``src/AnalysisG/events/gnn/cxx/particles.cxx``

**File Type**: C++ Source

**Lines**: 48

Description
-----------

particle_template* particle_gnn::clone(){return (particle_template*)new particle_gnn();}
void particle_gnn::build(std::map<std::string, particle_template*>* prt, element_t* el){

Dependencies
------------

**C++ Includes**:

- ``gnn-particles.h``

