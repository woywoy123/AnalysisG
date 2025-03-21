# distutils: language=c++
# cython: language_level=3

cdef dict elements(basic* null, data_t* data): 
    cdef basic vv
    if data.element(&vv): return {data.path : vv}
    return {}

# ------------------- (9.) Add the switch. And you are done =) --------------- #
cdef dict switch_board(data_t* data):
    if data.type == data_enum.vvf: return elements(<vector[vector[float ]]*>(NULL), data)
    if data.type == data_enum.vvd: return elements(<vector[vector[double]]*>(NULL), data)
    if data.type == data_enum.vvl: return elements(<vector[vector[long  ]]*>(NULL), data)
    if data.type == data_enum.vvi: return elements(<vector[vector[int   ]]*>(NULL), data)
    if data.type == data_enum.vvb: return elements(<vector[vector[bool  ]]*>(NULL), data)

    if data.type == data_enum.vf: return elements(<vector[float ]*>(NULL), data)
    if data.type == data_enum.vd: return elements(<vector[double]*>(NULL), data)
    if data.type == data_enum.vl: return elements(<vector[long  ]*>(NULL), data)
    if data.type == data_enum.vi: return elements(<vector[int   ]*>(NULL), data)
    if data.type == data_enum.vb: return elements(<vector[bool  ]*>(NULL), data)
    if data.type == data_enum.vc: return elements(<vector[char  ]*>(NULL), data)

    if data.type == data_enum.f: return elements(<float *>(NULL), data)
    if data.type == data_enum.d: return elements(<double*>(NULL), data)
    if data.type == data_enum.l: return elements(<long  *>(NULL), data)
    if data.type == data_enum.i: return elements(<int   *>(NULL), data)
    if data.type == data_enum.b: return elements(<bool  *>(NULL), data)
    if data.type == data_enum.c: return elements(<char  *>(NULL), data)

    if data.type == data_enum.ull: return elements(<unsigned long long*>(NULL), data)
    if data.type == data_enum.ui:  return elements(<unsigned int*>(NULL), data)
    return {}


