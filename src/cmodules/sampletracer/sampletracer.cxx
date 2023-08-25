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
        event_t* ev_ptr = &(event); 
        std::string event_r = ev_ptr -> event_root; 
        if (this -> root_map.count(event_r)){}
        else {this -> root_map[event_r] = new CyROOT(meta);}

        CyROOT* root = root_map[event_r];
        root -> AddEvent(ev_ptr);
        
        for (unsigned int x(0); x < code.size(); ++x)
        { 
            code_t* co = &(code[x]);
            ev_ptr -> code_hash.push_back(co -> hash);  
            if (this -> code_hashes.count(co -> hash)){ continue; }
            this -> code_hashes[co -> hash] = new Code::CyCode(); 
            this -> code_hashes[co -> hash] -> ImportCode(*co);
        }
    }

    tracer_t CySampleTracer::Export()
    {
        tracer_t output; 
        std::map<std::string, CyROOT*>::iterator itr;
        itr = this -> root_map.begin(); 
        for (; itr != this -> root_map.end(); ++itr)
        {
            std::string root_name = itr -> first; 
            CyROOT* r = itr -> second; 
            output.root_meta[root_name] = r -> meta; 
            output.root_names[root_name] = r -> Export(); 
        }
        return output; 
    }

    void CySampleTracer::Import(tracer_t inpt)
    {
        std::map<std::string, root_t>::iterator itr;
        itr = inpt.root_names.begin(); 
        for (; itr != inpt.root_names.end(); ++itr)
        {
            std::string root_name = itr -> first; 
            if (!this -> root_map.count(root_name))
            {
                meta_t* m = &(inpt.root_meta[root_name]); 
                this -> root_map[root_name] = new CyROOT(*m); 
            }
            CyROOT* root = this -> root_map[root_name]; 
            root -> Import(&(itr -> second)); 
        }

        std::map<std::string, code_t>::iterator itc; 
        itc = inpt.code.begin(); 
        for (; itc != inpt.code.end(); ++itc) 
        { 
            code_t* co = &(itc -> second); 
            if (this -> code_hashes.count(co -> hash)){ continue; }
            this -> code_hashes[co -> hash] = new Code::CyCode(); 
            this -> code_hashes[co -> hash] -> ImportCode(*co);
        }
    }


    std::vector<CyBatch*> CySampleTracer::MakeIterable()
    {
        std::vector<CyBatch*> output = {}; 
        std::map<std::string, CyROOT*>::iterator itr; 
        std::map<std::string, CyBatch*>::iterator itb; 
        itr = this -> root_map.begin(); 
        for (; itr != this -> root_map.end(); ++itr)
        {
            CyROOT* root = itr -> second; 
            itb = root -> batches.begin(); 
            for (; itb != root -> batches.end(); ++itb)
            {
                output.push_back(itb -> second); 
            }  
        }
        return output; 
    }


    std::map<std::string, int> CySampleTracer::length()
    {
        std::map<std::string, int> output; 
        std::map<std::string, int>::iterator itn; 
        std::map<std::string, CyROOT*>::iterator itr; 

        itr = this -> root_map.begin();
        for (; itr != this -> root_map.end(); ++itr)
        {
            CyROOT* r_ = itr -> second; 
            itn = r_ -> n_events.begin(); 
            for (; itn != r_ -> n_events.end(); ++itn)
            {
                output[itn -> first] += itn -> second; 
            }
        }
        return output; 
    }


}
