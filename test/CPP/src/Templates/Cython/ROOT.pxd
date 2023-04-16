from libcpp.string cimport string 
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "../Headers/ROOT.h" namespace "CyTracer":
    cdef cppclass CyEvent:
        CyEvent() except +

        string Tree; 
        string TrainMode; 
        string Hash; 
        string ROOT; 

        unsigned int EventIndex; 
        bool Graph;
        bool Event; 
        bool CachedEvent; 
        bool CachedGraph; 

        string ReturnROOTFile(); 
        string ReturnCachePath(); 

    cdef cppclass CyROOT:
        CyROOT() except +

        string Filename; 
        string SourcePath; 
        string CachePath; 

        bool operator==(CyROOT* root) except+
        CyROOT operator+(CyROOT* root) except+

    cdef cppclass CySampleTracer:
        CySampleTracer() except + 
         
        int length;  

        void AddEvent(CyEvent* event) except+
        map[string, bool] FastSearch(vector[string] hashes) except+

        # Converters
        vector[string] HashList() except+
            
        # Lookups 
        string HashToROOT(string Hash) except+
        vector[string] ROOTtoHashList(string root) except+
        bool ContainsROOT(string root) except+ 
        bool ContainsHash(string hash) except+ 

