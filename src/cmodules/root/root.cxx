#include "../root/root.h"

namespace SampleTracer
{

    CyBatch::CyBatch(std::string hash){this -> hash = hash;}
    CyBatch::~CyBatch()
    {
        std::map<std::string, CyEventTemplate*>::iterator ite; 
        ite = this -> events.begin(); 
        for(; ite != this -> events.end(); ++ite){delete ite -> second;} 

        std::map<std::string, CyGraphTemplate*>::iterator itg; 
        itg = this -> graphs.begin(); 
        for(; itg != this -> graphs.end(); ++itg){delete itg -> second;} 
 
        std::map<std::string, CySelectionTemplate*>::iterator its; 
        its = this -> selections.begin(); 
        for(; its != this -> selections.end(); ++its){delete its -> second;} 
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
    }


    void CyBatch::Import(const meta_t* meta)
    {
        if (this -> lock_meta){ return; }
        this -> meta = meta; 
        this -> lock_meta = true;
    }
 

    batch_t CyBatch::Export()
    {
        batch_t output; 
        
        std::map<std::string, CyEventTemplate*>::iterator ite; 
        std::map<std::string, CyGraphTemplate*>::iterator itg; 
        std::map<std::string, CySelectionTemplate*>::iterator its; 
        
        ite = this -> events.begin(); 
        itg = this -> graphs.begin();
        its = this -> selections.begin();
        output.hash = this -> hash;
         
        for (; ite != this -> events.end(); ++ite){
            output.events[ite -> first] = ite -> second -> Export(); 
        }

        for (; itg != this -> graphs.end(); ++itg){
            output.graphs[itg -> first] = itg -> second -> Export(); 
        }

        for (; its != this -> selections.end(); ++its){
            output.selections[its -> first] = its -> second -> selection; 
        }

        return output; 
    }

    void CyBatch::Export(batch_t* exp)
    {
        std::map<std::string, CyEventTemplate*>::iterator ite; 
        std::map<std::string, CyGraphTemplate*>::iterator itg; 
        std::map<std::string, CySelectionTemplate*>::iterator its; 
        
        ite = this -> events.begin(); 
        itg = this -> graphs.begin();
        its = this -> selections.begin();
        exp -> hash = this -> hash;
        
        for (; ite != this -> events.end(); ++ite){
            exp -> events[ite -> first] = ite -> second -> event; 
        }

        for (; itg != this -> graphs.end(); ++itg){
            exp -> graphs[itg -> first] = itg -> second -> Export(); 
        }

        for (; its != this -> selections.end(); ++its){
            exp -> selections[its -> first] = its -> second -> selection; 
        }
    }


    void CyBatch::Contextualize()
    {
        if (!this -> get_event){ this -> this_ev = nullptr; }
        if (!this -> get_graph){ this -> this_gr = nullptr; }
        if (!this -> get_selection){ this -> this_sel = nullptr; }

        std::string name = ""; 
        if (this -> this_tree.size()){ name += this -> this_tree; }
        if (this -> this_event_name.size()){ name += "/" + this -> this_event_name; }

        if (this -> events.count(name)){this -> this_ev = this -> events[name];}
        else { this -> this_ev = nullptr; }

        if (this -> graphs.count(name)){this -> this_gr = this -> graphs[name];}
        else { this -> this_gr = nullptr; }
        
        if (this -> selections.count(name)){this -> this_sel = this -> selections[name];}
        else { this -> this_sel = nullptr; } 

        if (this -> this_ev) { this -> valid = true; }
        if (this -> this_gr) { this -> valid = true; }
        if (this -> this_sel){ this -> valid = true; }
    }
    
    void CyBatch::ApplySettings(const settings_t* inpt)
    {
        this -> this_event_name = inpt -> eventname; 
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

    void CyBatch::ApplyCodeHash(const std::map<std::string, Code::CyCode*>* code_hash)
    {
        std::map<std::string, CyEventTemplate*>::iterator ite; 
        std::map<std::string, CyGraphTemplate*>::iterator itg; 
        std::map<std::string, CySelectionTemplate*>::iterator its; 

        ite = this -> events.begin(); 
        itg = this -> graphs.begin(); 
        its = this -> selections.begin(); 

        for (; ite != this -> events.end(); ++ite){
            CyEventTemplate* ev_ = ite -> second; 
            event_t* ev = &(ev_ -> event); 
            if (!code_hash -> count(ev -> code_hash)){ continue; }
            ev_ -> code_link = code_hash -> at(ev -> code_hash); 
        }

        std::map<std::string, std::string>::iterator it_s; 
        for (; itg != this -> graphs.end(); ++itg){
            CyGraphTemplate* gr_ = itg -> second;
            graph_t* gr = &(gr_ -> graph); 
            if (code_hash -> count(gr -> code_hash)){ 
                gr_ -> code_owner = false; 
                gr_ -> code_link = code_hash -> at(gr -> code_hash);
                continue;
            }
            
            if (code_hash -> count(gr_ -> topo_hash)){
                gr_ -> topo = code_hash -> at(gr_ -> topo_hash);
                continue;
            }
            
            it_s = gr -> edge_feature.begin(); 
            for (; it_s != gr -> edge_feature.end(); ++it_s){
                if (!code_hash -> count(it_s -> second)){ continue; }
                gr_ -> edge_fx[it_s -> first] = code_hash -> at(it_s -> second);
            }

            it_s = gr -> node_feature.begin(); 
            for (; it_s != gr -> node_feature.end(); ++it_s){
                if (!code_hash -> count(it_s -> second)){ continue; }
                gr_ -> node_fx[it_s -> first] = code_hash -> at(it_s -> second);
            }

            it_s = gr -> graph_feature.begin(); 
            for (; it_s != gr -> graph_feature.end(); ++it_s){
                if (!code_hash -> count(it_s -> second)){ continue; }
                gr_ -> graph_fx[it_s -> first] = code_hash -> at(it_s -> second);
            }

            it_s = gr -> pre_sel_feature.begin(); 
            for (; it_s != gr -> pre_sel_feature.end(); ++it_s){
                if (!code_hash -> count(it_s -> second)){ continue; }
                gr_ -> pre_sel_fx[it_s -> first] = code_hash -> at(it_s -> second);
            }
        }
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
        std::string event_h = event -> event_hash; 
        if (!this -> batches.count(event_h)){
            this -> batches[event_h] = new CyBatch(event_h);
        }

        CyBatch* bth = this -> batches[event_h];
        bth -> Import(&(this -> meta)); 
        bth -> Import(event); 
        std::string name = event -> event_tree; 
        name += "/" + event -> event_name; 
        this -> n_events[name] += 1; 
    }

    void CyROOT::AddGraph(const graph_t* graph)
    {
        std::string event_h = graph -> event_hash;
        if (!this -> batches.count(event_h)){
            this -> batches[event_h] = new CyBatch(event_h); 
        }

        CyBatch* bth = this -> batches[event_h]; 
        bth -> Import(&(this -> meta));
        bth -> Import(graph); 
        std::string name = graph -> event_tree; 
        name += "/" + graph -> event_name; 
        this -> n_graphs[name] += 1; 
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
        for (unsigned int x(0); x < this -> batches.size(); ++x)
        {
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
            itr = b -> events.begin(); 
            for (; itr != b -> events.end(); ++itr){
                this -> AddEvent(&(itr -> second)); 
            } 
        }
    }
}
