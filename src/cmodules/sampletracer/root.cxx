#include "../sampletracer/root.h"

namespace SampleTracer
{

    CyBatch::CyBatch(){}
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
    
    void CyBatch::Import(event_t event)
    { 
        std::string name = event.event_name; 
        if (this -> events.count(name)){ return; }
        CyEventTemplate* _event = new CyEventTemplate(); 
        _event -> Import(event); 
    }

    void CyBatch::Import(meta_t* meta)
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
            output.events[ite -> first] = ite -> second -> event; 
        }
        return output; 
    }




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

    void CyROOT::AddEvent(event_t event)
    {
        std::string event_h = event.event_hash; 
        bool contains = this -> batches.count(event_h);
        if (!contains){this -> batches[event_h] = new CyBatch();}
        CyBatch* bth = this -> batches[event_h];
        bth -> Import(&(this -> meta)); 
        bth -> Import(event); 
        this -> n_events[event.event_name] += 1; 
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
        output.meta = this -> meta; 
        output.n_events = this -> n_events; 
        output.n_graphs = this -> n_graphs; 
        output.n_selections = this -> n_selections;
        return output; 
    }

    void CyROOT::Import(root_t inpt)
    {



    }
}
