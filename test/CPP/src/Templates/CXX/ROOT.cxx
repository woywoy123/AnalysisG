#include "../Headers/ROOT.h"
#include "../Headers/Tools.h"

CyTracer::CyEvent::CyEvent(){}
CyTracer::CyROOT::CyROOT(){}
CyTracer::CySampleTracer::CySampleTracer(){}

CyTracer::CyEvent::~CyEvent(){}

CyTracer::CyROOT::~CyROOT(){}

CyTracer::CySampleTracer::~CySampleTracer(){}

std::string CyTracer::CyEvent::ReturnROOTFile()
{ 
    std::string Path = ""; 
    Path += this -> ROOTFile -> SourcePath + "/"; 
    Path += this -> ROOTFile -> Filename; 
    return Path; 
}

std::string CyTracer::CyEvent::ReturnCachePath()
{ 
    return this -> ROOTFile -> CachePath; 
}

std::vector<std::string> CyTracer::CyROOT::HashList()
{
    std::map<std::string, CyEvent*>::iterator it; 
    std::vector<std::string> out; 
    for (it = this -> HashMap.begin(); it != this -> HashMap.end(); ++it){ out.push_back(it -> first); }
    return out; 
}

void CyTracer::CySampleTracer::AddEvent(CyTracer::CyEvent* event)
{
    if (this -> _ROOTMap[event -> ROOT] == 0)
    {
        // Split the file path into a vector
        std::vector<std::string> val = Tools::Split( event -> ROOT, '/');
        
        // Constuct a new ROOT container object
        CyTracer::CyROOT* r = new CyTracer::CyROOT(); 

        // Get the filename i.e. <....>.root and remove it from the vector
        r -> Filename = val.back(); val.pop_back(); 
        
        // Extract the source directory
        for (std::string v : val){ r -> SourcePath += v + "/"; }
        
        // Add the ROOT filename to the collection
        this -> _ROOTMap[event -> ROOT] = r; 
    }
    
    // Link the Event object to the ROOT container 
    event -> ROOTFile = _ROOTMap[ event -> ROOT ]; 
    event -> ROOTFile -> length += 1; 

    // Link the Event hash to the ROOT file
    this -> _ROOTHash[event -> Hash] = event -> ROOTFile;  
      
    // Link the Event to the ROOT object
    event -> ROOTFile -> HashMap[event -> Hash] = event;
}

std::string CyTracer::CySampleTracer::HashToROOT(std::string Hash)
{
    if (this -> _ROOTHash[Hash] == 0){ return "None"; }
    CyROOT* r = this -> _ROOTHash[Hash]; 
    return (r -> SourcePath) + (r -> Filename); 
}

std::vector<std::string> CyTracer::CySampleTracer::HashList()
{
    std::map<std::string, CyROOT*>::iterator it; 
    std::vector<std::string> out; 
    for (it = this -> _ROOTHash.begin(); it != this -> _ROOTHash.end(); ++it){ out.push_back(it -> first); }
    this -> length = out.size(); 
    return out; 
}

std::vector<std::string> CyTracer::CySampleTracer::ROOTtoHashList(std::string root)
{
    if (this -> _ROOTMap[root] == 0){ return {}; }
    return this -> _ROOTMap[root] -> HashList(); 
}

bool CyTracer::CySampleTracer::ContainsROOT(std::string root)
{
    return (this -> ROOTtoHashList(root).size()) > 0; 
}

bool CyTracer::CySampleTracer::ContainsHash(std::string hash)
{
    return this -> HashToROOT(hash) != "None"; 
}

std::map<std::string, bool> CyTracer::CySampleTracer::FastSearch(std::vector<std::string> Hashes)
{
    auto search = [](std::vector<bool>* found, std::map<std::string, CyROOT*>* _hashes, std::vector<std::string>* hash)
    {
        for (int v(0); v < hash -> size(); ++v)
        {
            if (_hashes -> find(hash -> at(v)) == _hashes -> end()){ found -> push_back(false); }
            found -> push_back(true);  
        }
    }; 
    

    std::vector<std::vector<std::string>> q_hash = Tools::Chunk(Hashes, this -> ChunkSize); 
    std::vector<std::vector<bool>> q_found; 
    std::vector<std::thread*> tmp; 
    for (int v(0); v < q_hash.size(); ++v)
    {
        std::vector<bool> f = {}; 
        q_found.push_back(f);  
        std::thread* p = new std::thread(search, &q_found[v], &(this -> _ROOTHash), &q_hash[v]); 
        tmp.push_back(p); 
        
        if (tmp.size() < this -> Threads){ continue; }

        for (std::thread* t : tmp){ t -> join(); delete t; }
        tmp = {}; 
    }
    for (std::thread* t : tmp){ t -> join(); delete t; }
    tmp = {}; 

    std::map<std::string, bool> r; 
    for (int v(0); v < q_hash.size(); ++v)
    {
        for (int j(0); j < q_hash[v].size(); ++j)
        {
            r[q_hash[v][j]] = q_found[v][j]; 
        }
    }
    return r; 
}
