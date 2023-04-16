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
    auto search = [](bool* found, std::map<std::string, CyROOT*>* _hashes, std::string* hash)
    {
        if (_hashes -> find(*hash) == _hashes -> end()){ *found = false; }
        *found = true; 
    }; 

    std::map<std::string, bool> r; 
    std::vector<std::thread*> tmp; 
    for (std::string v : Hashes)
    { 
        r[v] = false; 

        std::thread* p = new std::thread(search, &r[v], &(this -> _ROOTHash), &v);
        tmp.push_back(p);

        // add threading limit....
    }
    
    for (std::thread* t : tmp){ t -> join(); delete t; }

    return r; 
}
