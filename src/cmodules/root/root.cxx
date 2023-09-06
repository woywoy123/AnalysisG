#include "../root/root.h"

namespace SampleTracer
{

    CyBatch::CyBatch(std::string hash){this -> hash = hash;}
    CyBatch::~CyBatch()
    {
        this -> destroy(&(this -> events)); 
        this -> destroy(&(this -> graphs)); 
        this -> destroy(&(this -> selections)); 
        if (!this -> code_hashes.size()){return;}
        std::map<std::string, Code::CyCode*>::iterator it; 
        it = this -> code_hashes.begin(); 
        for (; it != this -> code_hashes.end(); ++it){
            delete it -> second; 
        }
    }
    
    std::string CyBatch::Hash(){ return this -> hash; }
    void CyBatch::Import(const event_t* event)
    { 
        std::string name = ""; 
        name += event -> event_tree + "/"; 
        name += event -> event_name; 

        if (this -> events.count(name)){ return; }
        CyEventTemplate* _event = new CyEventTemplate(); 
        _event -> Import(*event); 
        this -> events[name] = _event; 
        this -> this_ev = _event; 
    }

    void CyBatch::Import(const graph_t* graph)
    {
        std::string name = ""; 
        name += graph -> event_tree + "/"; 
        name += graph -> event_name; 
      
        if (this -> graphs.count(name)){ return; }
        CyGraphTemplate* _graph = new CyGraphTemplate(); 
        _graph -> Import(*graph); 
        this -> graphs[name] = _graph; 
        this -> this_gr = _graph; 
    }


    void CyBatch::Import(const selection_t* selection)
    {
        std::string name = ""; 
        name += selection -> event_tree + "/"; 
        name += selection -> event_name; 
        
        if (this -> selections.count(name)){ return; }
        CySelectionTemplate* _selection = new CySelectionTemplate(); 
        _selection -> Import(*selection); 
        this -> selections[name] = _selection; 
        this -> this_sel = _selection; 
    }

    void CyBatch::Import(const meta_t* meta)
    {
        if (this -> lock_meta){ return; }
        this -> meta = meta; 
        this -> lock_meta = true;
    }

    void CyBatch::Import(const batch_t* bth)
    {
        std::map<std::string, event_t>::const_iterator ite;
        std::map<std::string, graph_t>::const_iterator itg;
        std::map<std::string, selection_t>::const_iterator its;
        ite = bth -> events.begin();
        itg = bth -> graphs.begin();
        its = bth -> selections.begin();
        
        for (; ite != bth -> events.end(); ++ite){ 
            this -> Import(&ite -> second); 
        }

        for (; itg != bth -> graphs.end(); ++itg){ 
            this -> Import(&itg -> second); 
        }

        for (; its != bth -> selections.end(); ++its){ 
            this -> Import(&its -> second); 
        }
        this -> hash = bth -> hash;
    }

    batch_t CyBatch::ExportPickled()
    {
        batch_t batch = this -> Export(); 
        this -> export_code(this -> events, &batch.code_hashes); 
        this -> export_code(this -> graphs, &batch.code_hashes); 
        this -> export_code(this -> selections, &batch.code_hashes); 
        return batch; 
    }

    void CyBatch::ImportPickled(const batch_t* inpt)
    {
        this -> Import(inpt);  
        const std::map<std::string, code_t>* code = &(inpt -> code_hashes);
        std::map<std::string, code_t>::const_iterator it; 
        std::map<std::string, Code::CyCode*> code_hashes = {}; 
        it = code -> begin();
        for (; it != code -> end(); ++it){
            code_hashes[it -> first] = new Code::CyCode();
            code_hashes[it -> first] -> ImportCode(it -> second, *code); 
        }
        this -> ApplyCodeHash(&code_hashes); 
        this -> code_hashes = code_hashes; 
    }

    batch_t CyBatch::Export()
    {
        batch_t output; 
        output.hash = this -> hash;
        if (this -> this_ev){ this -> export_this(this -> this_ev, &(output.events)); }
        else {this -> export_this(this -> events, &(output.events));}

        if (this -> this_gr){ this -> export_this(this -> this_gr, &(output.graphs)); }
        else {this -> export_this(this -> graphs, &(output.graphs));}

        if (this -> this_sel){ this -> export_this(this -> this_sel, &(output.selections)); }
        else {this -> export_this(this -> selections, &(output.selections));}
        return output; 
    }

    void CyBatch::Export(batch_t* exp)
    {
        exp -> hash = this -> hash;
        this -> export_this(this -> events, &(exp -> events));  
        this -> export_this(this -> graphs, &(exp -> graphs));  
        this -> export_this(this -> selections, &(exp -> selections));
    }

    void CyBatch::Contextualize()
    {
        std::string ev_name = ""; 
        std::string gr_name = "";
        std::string sel_name = ""; 
        if (this -> this_tree.size()){ 
            ev_name  = this -> this_tree + "/" + this -> this_event_name; 
            gr_name  = this -> this_tree + "/" + this -> this_graph_name; 
            sel_name = this -> this_tree + "/" + this -> this_selection_name;
        }
       
        if (this -> events.count(ev_name)){
            this -> this_ev = this -> events[ev_name];
        }
        else { this -> this_ev = nullptr; }

        if (this -> graphs.count(gr_name)){
            this -> this_gr = this -> graphs[gr_name];
        }
        else { this ->this_gr = nullptr; }
        
        if (this -> selections.count(sel_name)){
            this -> this_sel = this -> selections[sel_name];
        }
        else { this -> this_sel = nullptr; } 

        if (!this -> get_event){     this -> this_ev  = nullptr; }
        if (!this -> get_graph){     this -> this_gr  = nullptr; }
        if (!this -> get_selection){ this -> this_sel = nullptr; }

        if (this -> this_ev) { this -> valid = true; }
        if (this -> this_gr) { this -> valid = true; }
        if (this -> this_sel){ this -> valid = true; }
    }
    
    void CyBatch::ApplySettings(const settings_t* inpt)
    {
        this -> this_event_name = inpt -> eventname; 
        this -> this_graph_name = inpt -> graphname; 
        this -> this_selection_name = inpt -> selectionname; 
        this -> this_tree = inpt -> tree; 
        this -> get_event = inpt -> getevent; 
        this -> get_graph = inpt -> getgraph; 
        this -> get_selection = inpt -> getselection; 
        this -> valid = false; 
        this -> Contextualize();
        if (!this -> valid){ return; }

        const std::vector<std::string>* srch = &(inpt -> search); 
        unsigned int z = srch -> size(); 
        if (!z){ return; }
        
        for (unsigned int x(0); x < z; ++x){
            if (this -> hash != srch -> at(x)){continue;}
            return; 
        }
        
        for (unsigned int x(0); x < z; ++x){
            std::string find = srch -> at(x); 
            if (this -> meta -> original_input == find){ return; }
        }

        this -> valid = false; 
    }

    void CyBatch::LinkCode(
                    std::map<std::string, std::string>* inpt,
                    std::map<std::string, Code::CyCode*>* link,
                    const std::map<std::string, Code::CyCode*>* code_h)
    {
        std::map<std::string, std::string>::iterator it; 
        it = inpt -> begin(); 
        for (; it != inpt -> end(); ++it){
            if (!code_h -> count(it -> second)){continue;}
            Code::CyCode* co = code_h -> at(it -> second); 
            (*link)[it -> first] = co;
        }
    }

    void CyBatch::ApplyCodeHash(const std::map<std::string, Code::CyCode*>* code_hash)
    {

        this -> code_link(this -> events, code_hash); 
        this -> code_link(this -> graphs, code_hash); 
        this -> code_link(this -> selections, code_hash); 
        
        std::map<std::string, CyGraphTemplate*>::iterator itg; 
        itg = this -> graphs.begin(); 
        for (; itg != this -> graphs.end(); ++itg){
            CyGraphTemplate* gr_ = itg -> second;
            graph_t* gr = &(gr_ -> graph); 
            if (code_hash -> count(gr -> topo_hash)){
                gr_ -> topo_link = code_hash -> at(gr -> topo_hash);
                continue;
            }
            LinkCode(&(gr -> edge_feature), &(gr_ -> edge_fx), code_hash); 
            LinkCode(&(gr -> node_feature), &(gr_ -> node_fx), code_hash); 
            LinkCode(&(gr -> graph_feature), &(gr_ -> graph_fx), code_hash); 
            LinkCode(&(gr -> pre_sel_feature), &(gr_ -> pre_sel_fx), code_hash); 
        }
        this -> not_code_owner(&this -> graphs); 
    }

    CyROOT::CyROOT(meta_t meta){this -> meta = meta;}

    CyROOT::~CyROOT()
    {
        std::map<std::string, CyBatch*>::iterator it; 
        it = this -> batches.begin(); 
        for (; it != this -> batches.end(); ++it){delete it -> second;}
    }

    void CyROOT::AddEvent(const event_t* event)
    {
        std::map<std::string, event_t> inpt = {}; 
        inpt[event -> event_hash] = *event;  
        this -> ImportBatch(&inpt, &(this -> batches), &(this -> meta)); 
    }

    void CyROOT::AddGraph(const graph_t* graph)
    {
        std::map<std::string, graph_t> inpt = {}; 
        inpt[graph -> event_hash] = *graph;  
        this -> ImportBatch(&inpt, &(this -> batches), &(this -> meta)); 
    }

    void CyROOT::AddSelection(const selection_t* selection)
    {
        std::map<std::string, selection_t> inpt = {}; 
        inpt[selection -> event_hash] = *selection;
        this -> ImportBatch(&inpt, &(this -> batches), &(this -> meta)); 
    }

    void CyROOT::UpdateSampleStats()
    {
        this -> total_hashes = this -> batches.size();
        this -> n_events.clear(); 
        this -> n_graphs.clear(); 
        this -> n_selections.clear(); 

        std::map<std::string, CyEventTemplate*>::iterator ite; 
        std::map<std::string, CyGraphTemplate*>::iterator itg; 
        std::map<std::string, CySelectionTemplate*>::iterator its; 

        std::map<std::string, CyBatch*>::iterator bt = this -> batches.begin(); 
        for (; bt != this -> batches.end(); ++bt){
            CyBatch* this_b = bt -> second; 
            ite = this_b -> events.begin();
            itg = this_b -> graphs.begin();
            its = this_b -> selections.begin();

            for (; ite != this_b -> events.end(); ++ite){
                this -> n_events[ite -> first] += 1;
            }
            for (; itg != this_b -> graphs.end(); ++itg){
                this -> n_graphs[itg -> first] += 1;
            }
            for (; its != this_b -> selections.end(); ++its){
                this -> n_selections[its -> first] += 1;
            }
        }
    }

    std::map<std::string, std::vector<event_t*>> CyROOT::ReleaseEvents()
    {
        std::map<std::string, std::vector<event_t*>> out = {}; 
        std::map<std::string, CyBatch*>::iterator itr; 
        itr = this -> batches.begin(); 
        for (; itr != this -> batches.end(); ++itr){
            std::map<std::string, CyEventTemplate*> ev = itr -> second -> events; 
            std::map<std::string, CyEventTemplate*>::iterator ite; 
            ite = ev.begin(); 
            for (; ite != ev.end(); ++ite){
                event_t* ev_ = &(ite -> second -> event); 
                if (ev_ -> cached){continue;}
                std::string path = ev_ -> event_tree + "/" + ev_ -> event_name; 
                out[path].push_back(ev_); 
            }
        } 
        return out; 
    }

    root_t CyROOT::Export()
    {
        std::vector<batch_t*> container = {}; 
        std::vector<std::thread*> jobs = {}; 
        std::map<std::string, CyBatch*>::iterator it; 
        for (it = this -> batches.begin(); it != this -> batches.end(); ++it){
            CyBatch* exp = it -> second; 
            batch_t* get = new batch_t(); 
            
            std::thread* j = new std::thread(CyROOT::Make, exp, get); 
            container.push_back(get); 
            jobs.push_back(j); 
        }
       
        root_t output; 
        output.n_events = this -> n_events; 
        output.n_graphs = this -> n_graphs; 
        output.n_selections = this -> n_selections;
        for (unsigned int x(0); x < this -> batches.size(); ++x){
            jobs[x] -> join(); 
            batch_t* exp = container[x]; 

            output.batches[exp -> hash] = *exp; 
            delete exp; 
            delete jobs[x]; 
        }

        jobs.clear(); 
        container.clear();
        return output; 
    }

    void CyROOT::Import(const root_t* inpt)
    {
        std::map<std::string, event_t>::const_iterator itr; 
        std::map<std::string, batch_t>::const_iterator itb; 
        itb = inpt -> batches.begin();  
        for (; itb != inpt -> batches.end(); ++itb)
        {
            const batch_t* b = &(itb -> second); 
            std::string hash = b -> hash; 
            this -> ImportBatch(&(b -> events), &(this -> batches), &(this -> meta)); 
            this -> ImportBatch(&(b -> graphs), &(this -> batches), &(this -> meta)); 
            this -> ImportBatch(&(b -> selections), &(this -> batches), &(this -> meta)); 
        }
    }
}
