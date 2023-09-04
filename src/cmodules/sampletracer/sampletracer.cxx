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
        std::map<std::string, code_t> co = {}; 
        co[code.hash] = code; 
        CyHelpers::ImportCode(&(this -> code_hashes), &co); 
    }

    void CySampleTracer::AddEvent(event_t event, meta_t meta)
    {
        event_t* ev_ptr = &(event); 
        CyROOT* root = this -> AddContent(ev_ptr, &meta, &(this -> root_map)); 
        std::string event_name = ev_ptr -> event_name; 
        ev_ptr -> code_hash = this -> link_event_code[event_name]; 
        root -> AddEvent(ev_ptr);
    }

    void CySampleTracer::AddGraph(graph_t graph, meta_t meta)
    {
        graph_t* gr_ptr = &(graph);
        CyROOT* root = this -> AddContent(gr_ptr, &meta, &(this -> root_map)); 
        std::string event_name = gr_ptr -> event_name; 
        if (!this -> link_graph_code.count(event_name)){
            this -> link_graph_code[event_name] = gr_ptr -> code_hash;
        }
        gr_ptr -> code_hash = this -> link_graph_code[event_name]; 
        root -> AddGraph(gr_ptr); 
    }

    void CySampleTracer::AddSelection(selection_t selection, meta_t meta)
    {
        selection_t* sel_ptr = &(selection); 
        CyROOT* root = this -> AddContent(sel_ptr, &meta, &(this -> root_map)); 
        std::string event_name = sel_ptr -> event_name; 
        if (!this -> link_selection_code.count(event_name)){
            this -> link_selection_code[event_name] = sel_ptr -> code_hash;
        }
        sel_ptr -> code_hash = this -> link_selection_code[event_name]; 
        root -> AddSelection(sel_ptr); 
    }

    tracer_t CySampleTracer::Export()
    {
        tracer_t output; 
        std::map<std::string, CyROOT*>::iterator itr;
        itr = this -> root_map.begin(); 

        for (; itr != this -> root_map.end(); ++itr){
            std::string root_name = itr -> first; 
            CyROOT* r = itr -> second; 
            output.root_meta[root_name] = r -> meta; 
            output.root_names[root_name] = r -> Export(); 
        }
        output.link_event_code = this -> link_event_code; 
        output.event_trees = this -> event_trees;
        CyHelpers::ExportCode(&(output.hashed_code), this -> code_hashes); 
        return output; 
    }

    void CySampleTracer::Import(tracer_t inpt)
    {
        std::map<std::string, root_t>::iterator itr;
        itr = inpt.root_names.begin(); 

        for (; itr != inpt.root_names.end(); ++itr){
            std::string root_name = itr -> first; 
            if (!this -> root_map.count(root_name)){
                meta_t* m = &(inpt.root_meta[root_name]); 
                this -> root_map[root_name] = new CyROOT(*m); 
            }
            CyROOT* root = this -> root_map[root_name]; 
            root -> Import(&(itr -> second)); 
        }
        CyHelpers::ImportCode(&(this -> code_hashes), &(inpt.hashed_code)); 

        std::map<std::string, std::string>::iterator itl; 
        itl = inpt.link_event_code.begin(); 
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
        set.link_event_code = this -> link_event_code; 
        set.link_graph_code = this -> link_graph_code; 
        CyHelpers::ExportCode(&(set.hashed_code), code_hashes); 
        return set; 
    }

    void CySampleTracer::ImportSettings(settings_t inpt)
    {
        tracer_t tmp; 
        tmp.link_event_code = inpt.link_event_code; 
        tmp.link_graph_code = inpt.link_graph_code; 
        tmp.hashed_code = inpt.hashed_code; 

        inpt.hashed_code.clear(); 
        inpt.link_event_code.clear();
        inpt.link_graph_code.clear();
        this -> settings = inpt; 
        this -> Import(tmp); 
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
                CyHelpers::Make(ro, set, output[x], code); 
            }
            else {
                std::thread* j = new std::thread(CyHelpers::Make, ro, set, output[x], code);  
                jobs.push_back(j); 
            }
            ++x;
        }
        return CyHelpers::ReleaseVector(output, jobs, set -> threads); 
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

    std::map<std::string, int> CySampleTracer::length()
    {
        std::map<std::string, int> output; 
        std::map<std::string, int>::iterator itn; 
        std::map<std::string, CyROOT*>::iterator itr; 

        itr = this -> root_map.begin();
        for (; itr != this -> root_map.end(); ++itr){
            CyROOT* r_ = itr -> second;
            r_ -> UpdateSampleStats();  
            
            itn = r_ -> n_events.begin(); 
            for (; itn != r_ -> n_events.end(); ++itn){
                output[itn -> first] += itn -> second; 
            }

            itn = r_ -> n_graphs.begin(); 
            for (; itn != r_ -> n_graphs.end(); ++itn){
                output[itn -> first] += itn -> second; 
            }
            
            itn = r_ -> n_selections.begin(); 
            for (; itn != r_ -> n_selections.end(); ++itn){
                output[itn -> first] += itn -> second; 
            }
            output["n_hashes"] += r_ -> total_hashes; 
        }
        return output; 
    }



}
