node_features.h
===============

**File Path**: ``src/AnalysisG/graphs/ssml_mc20/include/ssml_mc20/node_features.h``

**File Type**: C++ Header

**Lines**: 45

Description
-----------

--------------------- Node Truth --------------------- //
void static res_node(int* o, particle_template* p){
if (type == "mu"){*o = ((muon*)p) -> from_res;}
else if (type == "el"){*o = ((electron*)p) -> from_res;}
else if (type == "jet"){*o = ((jet*)p) -> from_res;}

