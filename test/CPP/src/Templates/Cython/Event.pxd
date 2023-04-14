cdef extern from "../Headers/Tools.h" namespace "Tools":
    pass

cdef extern from "../Headers/Event.h" namespace "CyTemplate":
    cdef cppclass CyEvent:
        CyEvent() except +


