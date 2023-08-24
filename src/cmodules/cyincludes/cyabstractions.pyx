from libcpp.string cimport string
from cyabstractions cimport CyEvent
cdef string enc(str val): return val.encode("UTF-8")
cdef str env(string val): return val.decode("UTF-8")

cdef class Event:
    cdef CyEvent* ptr
    cdef meta_t* meta
    cdef event_t* ev

    def __cinit__(self): self.ptr = NULL
    def __init__(self):
        cdef str x = self.__class__.__name__
        self.ptr.add_eventname(enc(x))

    def __name__(self) -> str:
        return env(self.ptr.event.event_name)

    def __hash__(self) -> int: return int(self.hash[:8], 0)






