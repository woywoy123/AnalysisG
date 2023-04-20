#include "../Headers/ROOT.h"
#include "../Headers/Tools.h"

CyTracer::CyEvent::CyEvent(){}
CyTracer::CyROOT::CyROOT(){}
CyTracer::CySampleTracer::CySampleTracer(){}

CyTracer::CyEvent::~CyEvent(){}

CyTracer::CyROOT::~CyROOT()
{
    std::map<std::string, CyEvent*>::iterator it_e; 
    if ((this -> _Tracers).size() != 0){ return; }
    for (it_e = this -> HashMap.begin(); it_e != this -> HashMap.end(); ++it_e)
    {
        delete it_e -> second; 
    }
}

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

        // Add the tracer to the collection holding the ROOT pointer 
        r -> _Tracers.push_back(this); 
    }
    
    // Link the Event object to the ROOT container 
    event -> ROOTFile = _ROOTMap[ event -> ROOT ]; 
    event -> ROOTFile -> length += 1; 

    // Link the Event hash to the ROOT file
    this -> _ROOTHash[event -> Hash] = event -> ROOTFile;  
      
    // Link the Event to the ROOT object
    event -> ROOTFile -> HashMap[event -> Hash] = event;
}

CyTracer::CyEvent* CyTracer::CySampleTracer::HashToEvent(std::string Hash)
{
    CyTracer::CyROOT* root = this -> _ROOTHash[Hash]; 
    return root -> HashMap[Hash]; 
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
        for (unsigned int v(0); v < hash -> size(); ++v)
        {
            if (_hashes -> find(hash -> at(v)) == _hashes -> end()){ found -> push_back(false); }
            found -> push_back(true);  
        }
    }; 
    

    std::vector<std::vector<std::string>> q_hash = Tools::Chunk(Hashes, this -> ChunkSize); 
    std::vector<std::vector<bool>> q_found; 
    std::vector<std::thread*> tmp; 
    for (unsigned int v(0); v < q_hash.size(); ++v)
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
    for (unsigned int v(0); v < q_hash.size(); ++v)
    {
        for (unsigned int j(0); j < q_hash[v].size(); ++j)
        {
            r[q_hash[v][j]] = q_found[v][j]; 
        }
    }
    return r; 
}

// Operators 
bool CyTracer::CyEvent::operator==(CyEvent* p)
{
    if (this -> Hash != p -> Hash){ return false; } 
    if (this -> Graph != p -> Graph){ return false; }
    return true; 
}

bool CyTracer::CyROOT::operator==(CyROOT* p)
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

CyTracer::CyROOT* CyTracer::CyROOT::operator+(CyROOT* p)
{
    if (this == p){ return p; }
    CyROOT* r = new CyROOT(); 
    r -> Filename = this -> Filename; 
    r -> SourcePath = this -> SourcePath; 
    r -> CachePath = this -> CachePath; 
    r -> _Tracers = this -> _Tracers; 
    
    for (CySampleTracer* i : p -> _Tracers)
    { 
        bool same = false; 
        for (CySampleTracer* _k : r -> _Tracers){ if (_k == i){same = true; break;}}
        if (same){continue;}
        r -> _Tracers.push_back(i); 
    }

    r -> HashMap = this -> HashMap; 
    std::map<std::string, CyEvent*>::iterator it; 
    for (it = p -> HashMap.begin(); it != p -> HashMap.end(); ++it)
    { 
        std::string key = it -> first; 
        if (r -> HashMap[key] == 0)
        { 
            r -> HashMap[key] = it -> second; 
            continue; 
        }

        if (r -> HashMap[key] == p -> HashMap[key]){ continue; }
        r -> HashMap[key] = it -> second;  
    }
    return r; 
}

bool CyTracer::CySampleTracer::operator==(CySampleTracer* p)
{
    std::map<std::string, CyROOT*>::iterator it; 
    for (it = this -> _ROOTMap.begin(); it != this -> _ROOTMap.end(); ++it)
    {
        std::string key = it -> first; 
        if (p -> _ROOTMap[key] == 0){ return false; }
        if (p -> _ROOTMap[key] != it -> second){ return false; }
    }
    return true; 
}

CyTracer::CySampleTracer* CyTracer::CySampleTracer::operator+(CySampleTracer* p)
{
    if (this == p){ return p; }
    
    CySampleTracer* r = new CySampleTracer(); 
    r -> _ROOTMap = this -> _ROOTMap; 
    r -> _ROOTHash = this -> _ROOTHash; 
    r -> _ROOTHash.insert( p -> _ROOTHash.begin(), p -> _ROOTHash.end()); 

    std::map<std::string, CyROOT*>::iterator it; 
    for (it = p -> _ROOTMap.begin(); it != p -> _ROOTMap.end(); ++it)
    {
        std::string key = it -> first; 
        if (r -> _ROOTMap[key] == 0)
        {
            r -> _ROOTMap[key] = it -> second; 
            it -> second -> _Tracers.push_back(r); 
            continue; 
        }
        if (r -> _ROOTMap[key] == it -> second)
        { 
            r -> _ROOTMap[key] -> _Tracers.push_back(r); 
            continue; 
        }
        r -> _ROOTMap[key] = *(r -> _ROOTMap[key]) + it -> second; 
    }
    

    return r; 
}



