#include "../sampletracer/sampletracer.h"

namespace SampleTracer
{
    CySampleTracer::CySampleTracer(){}
    CySampleTracer::~CySampleTracer()
    {
        std::map<std::string, CyROOT*>::iterator it; 
        it = this -> root_map.begin();
        for (; it != this -> root_map.end(); ++it)
        {
            delete it -> second; 
        } 

        std::map<std::string, Code::CyCode*>::iterator itc; 
        itc = this -> code_hashes.begin();
        for (; itc != this -> code_hashes.end(); ++itc)
        {
            delete itc -> second; 
        }
    }

    void CySampleTracer::AddEvent(
            event_t event, 
            meta_t  meta, 
            std::vector<code_t> code)
    {
        std::string event_r = event.event_root; 
        if (this -> root_map.count(event_r)){}
        else {this -> root_map[event_r] = new CyROOT(meta);}

        CyROOT* root = root_map[event_r];
        root -> AddEvent(event);
        
        for (unsigned int x(0); x < code.size(); ++x)
        { 
            code_t* co = &(code[x]); 
            if (this -> code_hashes.count(co -> hash)){ continue; }
            this -> code_hashes[co -> hash] = new Code::CyCode(); 
            this -> code_hashes[co -> hash] -> ImportCode(*co);
        }
    }
}
