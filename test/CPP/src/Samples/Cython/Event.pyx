# distutils: language = c++

from libcpp.string cimport string
from Event cimport Event

cdef class CyEvent:
    cdef Event* ptr

    def __cinit__(self):
        self.ptr = new Event()

    def __dealloc__(self):
        del self.ptr
    
    @property
    def EventIndex(self):
        return self.ptr.EventIndex
    
    @EventIndex.setter
    def EventIndex(self, signed int val):
        self.ptr.EventIndex = val     
    
    @property
    def Hash(self):
        self.ptr.MakeHash()
        return self.ptr.Hash.decode("UTF-8")
    
    @Hash.setter
    def Hash(self, str value):
        self.ptr.Hash = <string>value.encode("UTF-8")
    
    @property
    def Compiled(self):
        return self.ptr.Compiled

    @property
    def Train(self):
        return self.ptr.Train


