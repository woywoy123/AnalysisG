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
        
        if (this -> graphs.count(name)){ return; }
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
        ite = this -> events.begin(); 
        for (; ite != this -> events.end(); ++ite)
        {
            std::string hash = ite -> first; 
            output.events[hash] = ite -> second -> event; 
        }
        return output; 
    }

    std::string CyBatch::Hash(){return this -> hash;}











    CyROOT::CyROOT(meta_t meta){this -> meta = meta;}

    CyROOT::~CyROOT()
    {
        std::map<std::string, CyBatch*>::iterator it; 
        it = this -> batches.begin(); 
        for (; it != this -> batches.end(); ++it)
        {
            delete it -> second; 
        }
    }

    void CyROOT::AddEvent(const event_t* event)
    {
        std::string event_h = event -> event_hash; 
        if (!this -> batches.count(event_h))
        {
            this -> batches[event_h] = new CyBatch(event_h);
        }

        CyBatch* bth = this -> batches[event_h];
        bth -> Import(&(this -> meta)); 
        bth -> Import(event); 
        this -> n_events[event -> event_name] += 1; 
    }

    root_t CyROOT::Export()
    {
        root_t output; 
        std::map<std::string, CyBatch*>::iterator it; 
        it = this -> batches.begin(); 
        for (; it != this -> batches.end(); ++it)
        {
            output.batches[it -> first] = it -> second -> Export();  
        }
        output.n_events = this -> n_events; 
        output.n_graphs = this -> n_graphs; 
        output.n_selections = this -> n_selections;
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
            for (; itr != b -> events.end(); ++b)
            {
                this -> AddEvent(&(itr -> second)); 
            } 
        }
    }
}
