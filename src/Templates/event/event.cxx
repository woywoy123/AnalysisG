#include <iostream>
#include "event.h"
#include "../tools/tools.h"

namespace CyTemplate
{
    CyEventTemplate::CyEventTemplate(){}
    CyEventTemplate::~CyEventTemplate(){}
        
    std::string CyEventTemplate::Hash(){return this -> event_hash;}

    void CyEventTemplate::Hash(std::string input)
    {
        if ((this -> event_hash).size() != 0){return;}
        this -> event_hash += "/" + ToString(this -> event_index); 
        this -> event_hash += "/" + this -> event_tree; 
        this -> event_hash = Hashing(this -> event_hash);
    }

    bool CyEventTemplate::operator==(CyEventTemplate* ev)
    {
        if (this -> event_hash != ev -> event_hash){ return false; }
        if (this -> event_name != ev -> event_name){ return false; }
        if (this -> event_tagging != ev -> event_tagging){ return false; }
        return true;  
    }

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
}
