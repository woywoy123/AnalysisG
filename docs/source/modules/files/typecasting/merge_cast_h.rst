merge_cast.h
============

**File Path**: ``modules/typecasting/include/tools/merge_cast.h``

**File Type**: H (Header)

**Lines**: 70

Dependencies
------------

**Includes**:

- ``map``
- ``string``
- ``vector``

Functions
---------

``void merge_data(std::vector<G>* out, std::vector<G>* p2)``

``void merge_data(G* out, G* p2)``

``void merge_data(std::map<g, G>* out, std::map<g, G>* p2)``

``void sum_data(G* out, G* p2)``

``void sum_data(std::vector<G>* out, std::vector<G>* p2)``

``void sum_data(std::map<g, G>* out, std::map<g, G>* p2)``

``void reserve_count(g* inp, long* ix)``

``void reserve_count(std::vector<g>* inp, long* ix)``

``void contract_data(std::vector<g>* out, g* p2)``

``void contract_data(std::vector<g>* out, std::vector<g>* p2)``

``void contract_data(std::vector<g>* out, std::vector<std::vector<g>>* p2)``

``void release_vector(std::vector<g>* ipt)``

