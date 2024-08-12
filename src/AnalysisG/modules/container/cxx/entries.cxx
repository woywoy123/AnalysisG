#include <container/container.h>


void entry_t::init(){
    this -> m_event     = new std::vector<event_template*>(); 
    this -> m_graph     = new std::vector<graph_template*>(); 
    this -> m_selection = new std::vector<selection_template*>(); 
    this -> m_data      = new std::vector<graph_t*>(); 
    
    this -> m_event -> reserve(10); 
    this -> m_graph -> reserve(10); 
    this -> m_selection -> reserve(10); 
    this -> m_data -> reserve(10); 
}

void entry_t::destroy(){
    if (this -> m_event){
        this -> destroy(this -> m_event);
        delete this -> m_event; 
        this -> m_event = nullptr; 
    }
    if (this -> m_graph){
        this -> destroy(this -> m_graph);
        delete this -> m_graph; 
        this -> m_graph = nullptr; 
    }
    if (this -> m_selection){
        this -> destroy(this -> m_selection);
        delete this -> m_selection; 
        this -> m_selection = nullptr; 
    }
}

bool entry_t::has_event(event_template* ev){
    std::string tr   = ev -> tree;
    std::string name = ev -> name; 
     
    for (size_t x(0); x < this -> m_event -> size(); ++x){
        std::string tr_   = (*this -> m_event)[x] -> tree; 
        std::string name_ = (*this -> m_event)[x] -> name; 
        if (tr_ == tr && name_  == name){return true;}
    }
    this -> m_event -> push_back(ev); 
    return false; 
}

bool entry_t::has_graph(graph_template* gr){
    std::string tr   = gr -> tree;
    std::string name = gr -> name; 
     
    for (size_t x(0); x < this -> m_graph -> size(); ++x){
        std::string tr_   = (*this -> m_graph)[x] -> tree; 
        std::string name_ = (*this -> m_graph)[x] -> name; 
        if (tr_ == tr && name_  == name){return true;}
    }
    this -> m_graph -> push_back(gr); 
    return false; 
}

bool entry_t::has_selection(selection_template* sel){
    std::string tr   = sel -> tree;
    std::string name = sel -> name; 
     
    for (size_t x(0); x < this -> m_selection -> size(); ++x){
        std::string tr_ = this -> m_selection -> at(x) -> tree; 
        std::string name_ = this -> m_selection -> at(x) -> name; 
        if (tr_ == tr && name_  == name){return true;}
    }
    this -> m_selection -> push_back(sel); 
    return false; 
}
