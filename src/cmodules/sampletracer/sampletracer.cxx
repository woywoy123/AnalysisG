#include "../sampletracer/sampletracer.h"

namespace SampleTracer
{
    CySampleTracer::CySampleTracer(){this -> settings = settings_t();}

    CySampleTracer::~CySampleTracer()
    {
        std::map<std::string, Code::CyCode*>::iterator itc; 
        std::map<std::string, CyROOT*>::iterator it; 

        itc = this -> code_hashes.begin();
        it = this -> root_map.begin();
        for (; it != this -> root_map.end(); ++it){delete it -> second;} 
        for (; itc != this -> code_hashes.end(); ++itc){delete itc -> second;}
    }

    void CySampleTracer::AddCode(code_t code)
    {
        if (this -> code_hashes.count(code.hash)){return;}
        this -> code_hashes[code.hash] = new Code::CyCode(); 
        this -> code_hashes[code.hash] -> ImportCode(code);  
    }

    void CySampleTracer::AddEvent(event_t event, meta_t meta)
    {
        event_t* ev_ptr = &(event); 
        std::string event_r = ev_ptr -> event_root; 
        if (!this -> root_map.count(event_r)){
            this -> root_map[event_r] = new CyROOT(meta);
        }
        CyROOT* root = root_map[event_r];

        std::string event_name = ev_ptr -> event_name; 
        ev_ptr -> code_hash = this -> link_event_code[event_name]; 
        root -> AddEvent(ev_ptr);
    }

    void CySampleTracer::AddGraph(graph_t graph, meta_t meta)
    {
        graph_t* gr_ptr = &(graph); 
        std::string event_r = gr_ptr -> event_root; 
        if (!this -> root_map.count(event_r)){
            this -> root_map[event_r] = new CyROOT(meta); 
        }
        CyROOT* root = root_map[event_r]; 

        std::string event_name = gr_ptr -> event_name; 
        gr_ptr -> code_hash = this -> link_event_code[event_name]; 
        root -> AddGraph(gr_ptr); 
    }

    tracer_t CySampleTracer::Export()
    {
        tracer_t output; 
        std::map<std::string, CyROOT*>::iterator itr;
        std::map<std::string, Code::CyCode*>::iterator itc; 
        itr = this -> root_map.begin(); 
        itc = this -> code_hashes.begin(); 

        for (; itr != this -> root_map.end(); ++itr){
            std::string root_name = itr -> first; 
            CyROOT* r = itr -> second; 
            output.root_meta[root_name] = r -> meta; 
            output.root_names[root_name] = r -> Export(); 
        }

        output.link_event_code = this -> link_event_code; 
        for (; itc != this -> code_hashes.end(); ++itc){
            output.hashed_code[itc -> first] = itc -> second -> ExportCode(); 
        }

        output.event_trees = this -> event_trees;
        return output; 
    }

    void CySampleTracer::Import(tracer_t inpt)
    {
        std::map<std::string, root_t>::iterator itr;
        std::map<std::string, code_t>::iterator itc; 
        std::map<std::string, std::string>::iterator itl; 
        itr = inpt.root_names.begin(); 
        itc = inpt.hashed_code.begin(); 
        itl = inpt.link_event_code.begin(); 

        for (; itr != inpt.root_names.end(); ++itr){
            std::string root_name = itr -> first; 
            if (!this -> root_map.count(root_name)){
                meta_t* m = &(inpt.root_meta[root_name]); 
                this -> root_map[root_name] = new CyROOT(*m); 
            }
            CyROOT* root = this -> root_map[root_name]; 
            root -> Import(&(itr -> second)); 
        }

        for (; itc != inpt.hashed_code.end(); ++itc){ 
            code_t* co = &(itc -> second); 
            if (this -> code_hashes.count(co -> hash)){ continue; }
            this -> code_hashes[co -> hash] = new Code::CyCode(); 
            this -> code_hashes[co -> hash] -> ImportCode(*co);
        }

        for (; itl != inpt.link_event_code.end(); ++itl){
            this -> link_event_code[itl -> first] = itl -> second; 
        }

        std::map<std::string, int>::iterator it_tr;
        it_tr = inpt.event_trees.begin(); 
        for (; it_tr != inpt.event_trees.end(); ++it_tr){
            std::string name = it_tr -> first; 
            this -> event_trees[name] += it_tr -> second;
        }
    }

    settings_t CySampleTracer::ExportSettings()
    {
        settings_t set = this -> settings; 
        std::map<std::string, Code::CyCode*>::iterator itc; 
        itc = this -> code_hashes.begin(); 
        for (; itc != this -> code_hashes.end(); ++itc){
            set.hashed_code[itc -> first] = itc -> second -> ExportCode(); 
        }
        set.link_event_code = this -> link_event_code; 
        return set; 
    }

    void CySampleTracer::ImportSettings(settings_t inpt)
    {
        tracer_t tmp; 
        tmp.link_event_code = inpt.link_event_code; 
        tmp.hashed_code = inpt.hashed_code; 

        inpt.hashed_code.clear(); 
        inpt.link_event_code.clear();
        this -> settings = inpt; 
        this -> Import(tmp); 
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
        for (; itr != this -> root_map.end(); ++itr){
            CyROOT* ro = itr -> second; 
            output.push_back(new std::vector<CyBatch*>()); 
            if (set -> threads == 1){ 
                CySampleTracer::Make(ro, set, output[x], code); ++x; 
                continue;
            }

            std::thread* j = new std::thread(CySampleTracer::Make, ro, set, output[x], code);  
            jobs.push_back(j); 
            ++x;
        }
        return this -> ReleaseVector(output, jobs); 
    }

    CySampleTracer* CySampleTracer::operator + (CySampleTracer* other)
    {
        CySampleTracer* smpl = new CySampleTracer(); 
        smpl -> Import(this -> Export()); 
        smpl -> Import(other -> Export()); 
        return smpl; 
    }

    void CySampleTracer::operator += (CySampleTracer* other){
        this -> Import(other -> Export());
    }

    void CySampleTracer::iadd(CySampleTracer* other){ 
        *this += other; 
    }

    std::map<std::string, int> CySampleTracer::length(){
        std::map<std::string, int> output; 
        std::map<std::string, int>::iterator itn; 
        std::map<std::string, CyROOT*>::iterator itr; 

        itr = this -> root_map.begin();
        for (; itr != this -> root_map.end(); ++itr){
            CyROOT* r_ = itr -> second; 
            itn = r_ -> n_events.begin(); 
            for (; itn != r_ -> n_events.end(); ++itn){
                output[itn -> first] += itn -> second; 
            }
        }
        return output; 
    }



}
