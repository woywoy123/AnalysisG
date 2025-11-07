gnn-event.h
===========

**File Path**: ``src/AnalysisG/events/gnn/include/inference/gnn-event.h``

**File Type**: C++ Header

**Lines**: 163

Description
-----------

void reduce(element_t* el, std::string key, g* out){
(*out) = tmp[0][0];
void reduce(element_t* el, std::string key, std::vector<g>* out, int dim){
if (dim == -1){(*out) = tmp[0]; return;}
(*out).push_back(tmp[x][0]);

Dependencies
------------

**C++ Includes**:

- ``inference/gnn-particles.h``
- ``templates/event_template.h``

Classes
-------

``gnn_event``
~~~~~~~~~~~~~

Class defined in this file.

