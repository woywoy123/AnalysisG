# distutils: language=c++
# cython: language_level=3

cdef dict elements(basic* null, data_t* data): 
    cdef basic vv
    if data.element(&vv): return {data.path : vv}
    return {}

# ------------------- (9.) Add the switch. And you are done =) --------------- #
cdef dict switch_board(data_t* data):
    if data.type == data_enum.vvv_f: return elements(<vector[vector[float ]]*>(NULL), data)
    if data.type == data_enum.vvv_d: return elements(<vector[vector[double]]*>(NULL), data)
    if data.type == data_enum.vvv_l: return elements(<vector[vector[long  ]]*>(NULL), data)
    if data.type == data_enum.vvv_i: return elements(<vector[vector[int   ]]*>(NULL), data)
    if data.type == data_enum.vvv_b: return elements(<vector[vector[bool  ]]*>(NULL), data)

    if data.type == data_enum.vv_f:  return elements(<vector[float ]*>(NULL), data)
    if data.type == data_enum.vv_d:  return elements(<vector[double]*>(NULL), data)
    if data.type == data_enum.vv_l:  return elements(<vector[long  ]*>(NULL), data)
    if data.type == data_enum.vv_i:  return elements(<vector[int   ]*>(NULL), data)
    if data.type == data_enum.vv_b:  return elements(<vector[bool  ]*>(NULL), data)
    if data.type == data_enum.vv_c:  return elements(<vector[char  ]*>(NULL), data)

    if data.type == data_enum.v_f:   return elements(<float *>(NULL), data)
    if data.type == data_enum.v_d:   return elements(<double*>(NULL), data)
    if data.type == data_enum.v_l:   return elements(<long  *>(NULL), data)
    if data.type == data_enum.v_i:   return elements(<int   *>(NULL), data)
    if data.type == data_enum.v_b:   return elements(<bool  *>(NULL), data)
    if data.type == data_enum.v_c:   return elements(<char  *>(NULL), data)

    if data.type == data_enum.v_ull: return elements(<unsigned long long*>(NULL), data)
    if data.type == data_enum.v_ui:  return elements(<unsigned int*>(NULL), data)
    return {}


