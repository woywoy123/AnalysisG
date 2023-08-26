#include "../sampletracer/sampletracer.h"

namespace SampleTracer
{
    CySampleTracer::CySampleTracer()
    {
        this -> settings = settings_t(); 
    }

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

    void CySampleTracer::AddEvent(event_t event, meta_t  meta, std::vector<code_t> code)
    {
        event_t* ev_ptr = &(event); 
        std::string event_r = ev_ptr -> event_root; 
        if (this -> root_map.count(event_r)){}
        else {this -> root_map[event_r] = new CyROOT(meta);}

        for (unsigned int x(0); x < code.size(); ++x)
        { 
            code_t* co = &(code[x]);
            ev_ptr -> code_hash.push_back(co -> hash);  
            if (this -> code_hashes.count(co -> hash)){ continue; }
            this -> code_hashes[co -> hash] = new Code::CyCode(); 
            this -> code_hashes[co -> hash] -> ImportCode(*co);
        }

        CyROOT* root = root_map[event_r];
        root -> AddEvent(ev_ptr);
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

    std::vector<CyBatch*> CySampleTracer::ReleaseVector(
            std::vector<std::vector<CyBatch*>*> output, 
            std::vector<std::thread*> jobs)
    {
        std::vector<CyBatch*> release = {}; 
        settings_t* set = &(this -> settings); 

        for (unsigned int x(0); x < output.size(); ++x)
        {
            if (set -> threads != 1){ jobs[x] -> join(); }
            release.insert(release.end(), output[x] -> begin(), output[x] -> end()); 
            output[x] -> clear(); 
            delete output[x]; 
            if (set -> threads != 1){ delete jobs[x]; }
        }
        return release; 
    }

    std::vector<CyBatch*> CySampleTracer::MakeIterable()
    {

        unsigned int x = 0; 
        std::vector<std::thread*> jobs = {};  
        settings_t* set = &(this -> settings);
        
        std::vector<std::vector<CyBatch*>*> output = {}; 
        std::map<std::string, Code::CyCode*>* code = &(this -> code_hashes);   

        std::map<std::string, CyROOT*>::iterator itr = this -> root_map.begin(); 

        for (; itr != this -> root_map.end(); ++itr)
        {
            CyROOT* ro = itr -> second; 
            output.push_back(new std::vector<CyBatch*>()); 

            if (set -> threads == 1){ CySampleTracer::Make(ro, set, output[x], code); ++x; continue;}
            std::thread* j = new std::thread(CySampleTracer::Make, ro, set, output[x], code);  
            jobs.push_back(j); 
            ++x;
        }
        return this -> ReleaseVector(output, jobs); 
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
