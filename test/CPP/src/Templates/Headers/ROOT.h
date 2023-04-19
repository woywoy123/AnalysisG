#ifndef __ROOT__H
#define __ROOT__H

#include <iostream> 
#include <map> 
#include <vector>
#include <thread>
#include <future>

namespace CyTracer
{
    class CyEvent; 
    class CyROOT; 
    class CySampleTracer; 

    class CyEvent
    {
        public:
            // Constructor 
            CyEvent(); 

            // Destructor 
            ~CyEvent();

            // Operators 
            bool operator==(CyEvent* p); 

            // Event Definitions 
            std::string Tree = ""; 
            std::string TrainMode = "";
            std::string Hash = "";
            std::string ROOT = ""; 
            
            unsigned int EventIndex = 0; 
            bool Graph = false; 
            bool Event = false; 
            bool CachedEvent = false; 
            bool CachedGraph = false; 

            std::string ReturnROOTFile(); 
            std::string ReturnCachePath(); 

            CyROOT* ROOTFile; 
    }; 

    class CyROOT
    {
        public: 
            // Constructor 
            CyROOT(); 

            // Destructor 
            ~CyROOT(); 

            std::string Filename = ""; 
            std::string SourcePath = ""; 
            std::string CachePath = ""; 
            int length = 0; 
        
            std::map<std::string, CyEvent*> HashMap;

            // Functions 
            std::vector<std::string> HashList(); 

            // Book keeping
            std::vector<CySampleTracer*> _Tracers = {};
            
            // Operators
            bool operator==(CyROOT* p); 
            CyROOT* operator+(CyROOT* p); 
    }; 

    class CySampleTracer
    {
        public:
            // Constructor 
            CySampleTracer(); 
            
            // Destructor 
            ~CySampleTracer(); 
            
            void AddEvent(CyEvent* event); 

            // Converters
            std::vector<std::string> HashList();
            
            // Lookups 
            std::string HashToROOT(std::string Hash);
            std::vector<std::string> ROOTtoHashList(std::string root); 
            std::map<std::string, bool> FastSearch(std::vector<std::string> Hashes); 
            bool ContainsROOT(std::string root); 
            bool ContainsHash(std::string hash); 
            int length = 0; 
            int Threads = 1;
            int ChunkSize = 100; 

            // Operators 
            bool operator==(CySampleTracer* p); 
            CySampleTracer* operator+(CySampleTracer* p); 

        private:
            std::map<std::string, CyROOT*> _ROOTMap = {}; 
            std::map<std::string, CyROOT*> _ROOTHash = {}; 
     }; 
}
#endif