graph_features.h
================

**File Path**: ``src/AnalysisG/graphs/ssml_mc20/include/ssml_mc20/graph_features.h``

**File Type**: C++ Header

**Lines**: 76

Description
-----------

------------------ Truth Graph Features --------------------- //
void static signal_event(bool* o, ssml_mc20* event){
if (event -> Tops.size() != 4){*o = false; return;}
std::vector<particle_template*> t = event -> Tops;
top* t_ = (top*)t[x];

