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
            bool operator==(CyEvent* p)
            {
                if (this -> Hash != p -> Hash){ return false; } 
                if (this -> Graph != p -> Graph){ return false; } 
                return true; 
            }

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
            
            bool operator==(CyROOT* p)
            {
                if (this -> Filename != p -> Filename){ return false; }            
                if (this -> SourcePath != p -> SourcePath){ return false; }            
                if (this -> length != p -> length){ return false; }
                std::map<std::string, CyEvent*>::iterator it; 
                for (it = this -> HashMap.begin(); it != this -> HashMap.end(); ++it)
                {
                    std::string hash = it -> first; 
                    if (p -> HashMap[hash] == 0){ return false; }
                    if (p -> HashMap[hash] != it -> second){ return false; }
                }
                return true;  
            }

            CyROOT operator+(CyROOT* p)
            {
                if (this == p){ return (*this); }
                CyTracer::CyROOT R; 
                R.Filename = this -> Filename;  
                R.SourcePath = this -> SourcePath; 
                R.CachePath = this -> CachePath; 
                R.HashMap = this -> HashMap; 
                R.length = this -> length; 
                 
                std::map<std::string, CyEvent*>::iterator it; 
                for (it = p -> HashMap.begin(); it != p -> HashMap.begin(); ++it)
                {
                    if (R.HashMap[it -> first] != 0){continue;}
                    R.HashMap[it -> first] = it -> second;
                    R.length += 1; 
                }
                return R; 
            }
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

        private:
            std::map<std::string, CyROOT*> _ROOTMap = {}; 
            std::map<std::string, CyROOT*> _ROOTHash = {}; 
     }; 
}
#endif
