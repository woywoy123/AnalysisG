structs.pyx
===========

**File Path**: ``src/AnalysisG/core/structs.pyx``

**File Type**: Cython Source

**Lines**: 36

Description
-----------

cdef dict elements(basic* null, data_t* data):
cdef dict switch_board(data_t* data):
if data.type == data_enum.vvv_f: return elements(<vector[vector[float ]]*>(NULL), data)
if data.type == data_enum.vvv_d: return elements(<vector[vector[double]]*>(NULL), data)
if data.type == data_enum.vvv_l: return elements(<vector[vector[long  ]]*>(NULL), data)

