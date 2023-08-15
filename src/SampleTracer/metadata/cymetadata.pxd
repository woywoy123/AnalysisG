from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "metadata.h" namespace "SampleTracer":
    cdef cppclass CyMetaData:
        CyMetaData() except +

        string AMITag
        string generators
        unsigned int dsid

        void addsamples(int index, string sample) except +
        void addconfig(string key, string val) except +

        bool isMC
        string derivationFormat
        map[int, string] inputfiles
        map[string, string] config
        unsigned int eventNumber

