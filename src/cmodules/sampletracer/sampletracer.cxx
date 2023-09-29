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

    void CySampleTracer::AddMeta(meta_t meta, std::string event_root)
    {
        if (!this -> root_map.count(event_root)){
            this -> root_map[event_root] = new CyROOT(meta);
        }
        CyROOT* r = this -> root_map[event_root]; 
        r -> meta.sample_name = meta.sample_name; 
    }

    CyBatch* CySampleTracer::RegisterHash(std::string hash, std::string event_root)
    {
        CyROOT* root = this -> root_map[event_root]; 
        if (root -> batches.count(hash)){return root -> batches[hash];}
        CyBatch* batch = new CyBatch(hash); 
        batch -> Import(&(root -> meta));
        root -> batches[hash] = batch;
        return root -> batches[hash]; 
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

    void CySampleTracer::AddCode(code_t code)
    {
        std::map<std::string, code_t> co = {}; 
        co[code.hash] = code; 
        CyHelpers::ImportCode(&(this -> code_hashes), &co); 
    }

    std::map<std::string, std::vector<CyEventTemplate*>> CySampleTracer::DumpEvents()
    {
        std::map<std::string, std::vector<CyEventTemplate*>> output; 
        this -> ReleaseObjects(&output);
        return output;
    }

    std::map<std::string, std::vector<CyGraphTemplate*>> CySampleTracer::DumpGraphs()
    {
        std::map<std::string, std::vector<CyGraphTemplate*>> output; 
        this -> ReleaseObjects(&output);
        return output;
    }

    std::map<std::string, std::vector<CySelectionTemplate*>> CySampleTracer::DumpSelections()
    {
        std::map<std::string, std::vector<CySelectionTemplate*>> output = {}; 
        this -> ReleaseObjects(&output);
        return output;
    }

    void CySampleTracer::FlushEvents(std::vector<std::string> hashes)
    {
        bool tmp = this -> settings.getevent;
        this -> settings.search = hashes;  
        this -> settings.getevent = true; 
        std::vector<CyBatch*> smpls = this -> MakeIterable(); 
        for (unsigned int i(0); i < smpls.size(); ++i){
            std::string root_name = smpls[i] -> this_ev -> event.event_root; 
            std::string name = this -> make_flush_string(root_name, "EventCache"); 
            smpls[i] -> event_dir[name] = root_name; 
            smpls[i] -> destroy(&smpls[i] -> events); 
        }
        this -> settings.search.clear(); 
        this -> settings.getevent = tmp;
    }


    void CySampleTracer::FlushGraphs(std::vector<std::string> hashes)
    {
        bool tmp = this -> settings.getgraph;
        this -> settings.search = hashes;  
        this -> settings.getgraph = true; 
        std::vector<CyBatch*> smpls = this -> MakeIterable(); 
        for (unsigned int i(0); i < smpls.size(); ++i){
            std::string root_name = smpls[i] -> this_gr -> graph.event_root; 
            std::string name = this -> make_flush_string(root_name, "GraphCache"); 
            smpls[i] -> graph_dir[name] = root_name; 
            smpls[i] -> destroy(&smpls[i] -> graphs); 
        }
        this -> settings.search.clear(); 
        this -> settings.getgraph = tmp;
    }

    void CySampleTracer::FlushSelections(std::vector<std::string> hashes)
    {
        bool tmp = this -> settings.getselection;
        this -> settings.search = hashes;  
        this -> settings.getselection = true; 
        std::vector<CyBatch*> smpls = this -> MakeIterable(); 
        for (unsigned int i(0); i < smpls.size(); ++i){
            std::string root_name = smpls[i] -> this_sel -> selection.event_root; 
            std::string name = this -> make_flush_string(root_name, "SelectionCache"); 
            smpls[i] -> selection_dir[name] = root_name; 
            smpls[i] -> destroy(&smpls[i] -> selections); 
        }
        this -> settings.search.clear(); 
        this -> settings.getselection = tmp;
    }

    void CySampleTracer::DumpTracer()
    {
        export_t* out = &(this -> state); 
        std::map<std::string, CyROOT*>::iterator itr;
        itr = this -> root_map.begin(); 

        for (; itr != this -> root_map.end(); ++itr){
            CyROOT* r = itr -> second; 
            std::string root_name = itr -> first; 
            out -> root_meta[root_name] = r -> meta; 
        }
        out -> link_event_code = this -> link_event_code; 
        out -> link_graph_code = this -> link_graph_code; 
        out -> link_selection_code = this -> link_selection_code; 
        CyHelpers::ExportCode(&(out -> hashed_code), this -> code_hashes); 
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
        output.link_graph_code = this -> link_graph_code; 
        output.link_selection_code = this -> link_selection_code; 
        output.event_trees = this -> event_trees;
        CyHelpers::ExportCode(&(output.hashed_code), this -> code_hashes); 
        return output; 
    }

    void CySampleTracer::Import(tracer_t inpt)
    {
        auto add = [](
                std::map<std::string, std::string>* inpt, 
                std::map<std::string, std::string>* toadd)
        {
            std::map<std::string, std::string>::iterator itl;
            itl = inpt -> begin(); 
            for (; itl != inpt -> end(); ++itl){
                if (toadd -> count(itl -> first)){continue;}
                (*toadd)[itl -> first] = itl -> second;
            }
        };

        std::map<std::string, root_t>::iterator itr;
        itr = inpt.root_names.begin(); 

        for (; itr != inpt.root_names.end(); ++itr){
            std::string root_name = itr -> first; 
            if (!this -> root_map.count(root_name)){
                meta_t* m = &(inpt.root_meta[root_name]); 
                this ->root_map[root_name] = new CyROOT(*m); 
            }
            CyROOT* root = this -> root_map[root_name]; 
            root -> Import(&(itr -> second)); 
        }
        CyHelpers::ImportCode(&(this -> code_hashes), &(inpt.hashed_code)); 
        add(&(inpt.link_event_code), &this -> link_event_code);
        add(&(inpt.link_graph_code), &this -> link_graph_code);
        add(&(inpt.link_selection_code), &this -> link_selection_code);

        std::map<std::string, int>::iterator it_tr = inpt.event_trees.begin(); 
        for (; it_tr != inpt.event_trees.end(); ++it_tr){
            this -> event_trees[it_tr -> first] += it_tr -> second;
        }
    }

    settings_t CySampleTracer::ExportSettings()
    {
        settings_t set = this -> settings; 
        set.link_event_code = this -> link_event_code; 
        set.link_graph_code = this -> link_graph_code; 
        set.link_selection_code = this -> link_selection_code; 
        CyHelpers::ExportCode(&(set.hashed_code), code_hashes); 
        return set; 
    }

    void CySampleTracer::ImportSettings(settings_t inpt)
    {
        tracer_t tmp; 
        tmp.link_event_code = inpt.link_event_code; 
        tmp.link_graph_code = inpt.link_graph_code; 
        tmp.link_selection_code = inpt.link_selection_code; 
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
        smpl -> ImportSettings(this -> ExportSettings()); 
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
        auto populate = [](
                std::map<std::string, int>* inpt, 
                std::map<std::string, int>* out)
        {
            std::map<std::string, int>::iterator its; 
            its = inpt -> begin(); 
            for (; its != inpt -> end(); ++its){
                (*out)[its -> first] += its -> second; 
            }
        };


        std::map<std::string, int> output = {}; 
        std::map<std::string, CyROOT*>::iterator itr; 
        this -> event_trees.clear(); 

        output["n_hashes"] = 0; 
        itr = this -> root_map.begin();
        for (; itr != this -> root_map.end(); ++itr){
            CyROOT* r_ = itr -> second;
            r_ -> UpdateSampleStats();  
            populate(&(r_ -> n_events), &output);
            populate(&(r_ -> n_graphs), &output); 
            populate(&(r_ -> n_selections), &output); 
            populate(&(r_ -> event_trees), &(this -> event_trees)); 
            output["n_hashes"] += r_ -> total_hashes; 
        }
        return output; 
    }



}
