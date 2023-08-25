#include "../event/event.h"

namespace CyTemplate
{
    CyEventTemplate::CyEventTemplate(){}
    CyEventTemplate::~CyEventTemplate(){}
        
    void CyEventTemplate::addleaf(std::string key, std::string leaf)
    {
        this -> leaves[key] = leaf; 
    }

    void CyEventTemplate::addbranch(std::string key, std::string branch)
    {
        this -> branches[key] = branch; 
    }

    void CyEventTemplate::addtree(std::string key, std::string tree)
    {
        this -> trees[key] = tree; 
    }

    event_T CyEventTemplate::Export()
    {
        event_T tmp; 
        tmp.leaves = this -> leaves; 
        tmp.branches = this -> branches; 
        tmp.trees = this -> trees; 

        tmp.event = this -> event; 
        tmp.event.event = true; 

        tmp.meta = this -> meta; 
        return tmp;         
    }

    void CyEventTemplate::Import(event_T event)
    {
        this -> leaves = event.leaves; 
        this -> branches = event.branches; 
        this -> trees = event.trees; 

        this -> event = event.event; 
        this -> event.event = true; 
        this -> meta = event.meta; 
    }

    void CyEventTemplate::Import(event_t event)
    {
        this -> event = event; 
        this -> event.event = true;
    }

    bool CyEventTemplate::operator == (CyEventTemplate& ev)
    {
        event_t* ev1 = &(this -> event); 
        event_t* ev2 = &(ev.event); 
        if (ev1 -> event_hash    != ev2 -> event_hash   ){ return false; }
        if (ev1 -> event_name    != ev2 -> event_name   ){ return false; }
        if (ev1 -> event_tree    != ev2 -> event_tree   ){ return false; }
        if (ev1 -> event_tagging != ev2 -> event_tagging){ return false; }
        return true;  
    }



    CyGraphTemplate::CyGraphTemplate(){}
    CyGraphTemplate::~CyGraphTemplate(){}
    void CyGraphTemplate::Import(graph_t gr)
    {
        this -> graph = gr; 
        this -> graph.graph = true; 
    }; 


    CySelectionTemplate::CySelectionTemplate(){}
    CySelectionTemplate::~CySelectionTemplate(){}
    void CySelectionTemplate::Import(selection_t sel)
    {
        this -> selection = sel; 
        this -> selection.selection = true; 
    }; 





}
