node_features.h
===============

**File Path**: ``src/AnalysisG/graphs/exp_mc20/include/exp_mc20/node_features.h``

**File Type**: C++ Header

**Lines**: 46

Description
-----------

--------------------- Node Truth --------------------- //
void static res_node(int* o, particle_template* p){
std::string type = p -> type;
if (type == "mu"){*o = ((muon*)p) -> from_res;}
else if (type == "el"){*o = ((electron*)p) -> from_res;}

