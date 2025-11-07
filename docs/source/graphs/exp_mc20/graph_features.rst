graph_features.h
================

**File Path**: ``src/AnalysisG/graphs/exp_mc20/include/exp_mc20/graph_features.h``

**File Type**: C++ Header

**Lines**: 62

Description
-----------

------------------ Truth Graph Features --------------------- //
void static signal_event(bool* o, exp_mc20* event){
if (event -> Tops.size() != 4){*o = false; return;}
std::vector<particle_template*> t = event -> Tops;
for (size_t x(0); x < t.size(); ++x){

