from libcpp.string cimport string 
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "../Headers/ROOT.h" namespace "CyTracer":
    cdef cppclass CyEvent:
        CyEvent() except +

        string Tree; 
        string TrainMode; 
        string hash; 
        string ROOT; 

        unsigned int EventIndex; 
        bool Graph;
        bool Event; 
        bool CachedEvent; 
        bool CachedGraph;

        string ReturnROOTFile(); 
        string ReturnCachePath();
        void Hash(); 
        CyROOT* ROOTFile; 

    cdef cppclass CyROOT:
        CyROOT() except +

        string Filename; 
        string SourcePath; 
        string CachePath; 
        map[string, CyEvent*] HashMap; 
        vector[CySampleTracer*] _Tracers; 

        bool operator==(CyROOT* root) except+
        CyROOT operator+(CyROOT* root) except+

    cdef cppclass CySampleTracer:
        CySampleTracer() except + 
         
        int length;
        int Threads; 
        int ChunkSize; 
        map[string, CyROOT*] _ROOTMap; 
        map[string, CyROOT*] _ROOTHash; 

        void AddEvent(CyEvent* event) except+
        map[string, bool] FastSearch(vector[string] hashes) except+

        # Converters
        vector[string] HashList() except+
            
        # Lookups 
        CyEvent* HashToEvent(string Hash) except +
        string HashToROOT(string Hash) except +
        vector[string] ROOTtoHashList(string root) except +
        bool ContainsROOT(string root) except + 
        bool ContainsHash(string hash) except + 

        bool operator==(CySampleTracer* smple) except +
        CySampleTracer* operator+(CySampleTracer* smple) except + 
