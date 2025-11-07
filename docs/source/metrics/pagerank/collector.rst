collector.cxx
=============

**File Path**: ``src/AnalysisG/metrics/pagerank/cxx/collector.cxx``

**File Type**: C++ Source

**Lines**: 85

Description
-----------

void collector::add_truth(kinematic_t* p, std::string mode, std::string model, int kfold, int epoch){
this -> data[model][epoch][kfold].data[mode][this -> idx].truth.push_back(*p);
void collector::add_pagerank(kinematic_t* p, std::string mode, std::string model, int kfold, int epoch){

Dependencies
------------

**C++ Includes**:

- ``metrics/pagerank.h``

