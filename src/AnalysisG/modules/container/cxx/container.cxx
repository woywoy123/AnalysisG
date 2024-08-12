#include <container/container.h>

container::container(){
    this -> random_access = new std::map<std::string, entry_t*>(); 
}

container::~container(){
    delete this -> meta_data; 
    delete this -> filename;  
    delete this -> label; 

    std::map<std::string, entry_t*>::iterator itr; 
    itr = this -> random_access -> begin();
    for (; itr != this -> random_access -> end(); ++itr){
        entry_t* ev = itr -> second; 
        ev -> destroy(); 
        delete ev -> hash; 
        delete ev; 
    }
    this -> random_access -> clear(); 
    delete this -> random_access; 

    if (!this -> merged){return;}
    std::map<std::string, selection_template*>::iterator itrs;
    itrs = this -> merged -> begin(); 
    for (; itrs != this -> merged -> end(); ++itrs){delete itrs -> second;}
    delete this -> merged; 
}

void container::get_events(std::vector<event_template*>* out, std::string label){
    if (label != *this -> label && label.size()){return;}
    std::map<std::string, entry_t*>::iterator itr = this -> random_access -> begin(); 
    for (; itr != this -> random_access -> end(); ++itr){
        entry_t* ev = itr -> second; 
        std::vector<event_template*>* rn = ev -> m_event; 
        out -> insert(out -> end(), rn -> begin(), rn -> end()); 
    } 
}

void container::add_meta_data(meta* data, std::string fname){
    this -> filename = new std::string(fname); 
    this -> meta_data = data; 
}

meta* container::get_meta_data(){return this -> meta_data;}

entry_t* container::add_entry(std::string hash){
    entry_t* t = nullptr; 
    if (!this -> random_access -> count(hash)){
        t = new entry_t();
        t -> init(); 
        t -> hash = new std::string(hash); 
        t -> meta_data = this -> meta_data; 
        (*this -> random_access)[hash] = t; 
    }
    else {t = (*this -> random_access)[hash];}
    return t; 
}

bool container::add_event_template(event_template* ev, std::string _label, long* alloc){
    if (!this -> label){this -> label = new std::string(_label);}
    if (!this -> alloc){this -> alloc = *alloc;}

    std::string hash = ev -> hash; 
    entry_t* evt = this -> add_entry(hash); 
    return evt -> has_event(ev); 
}

bool container::add_graph_template(graph_template* gr, std::string _label){
    if (!this -> label){this -> label = new std::string(_label);}

    std::string hash = gr -> hash; 
    entry_t* evt = this -> add_entry(hash); 
    return evt -> has_graph(gr); 
}

bool container::add_selection_template(selection_template* sel){
    std::string hash = sel -> hash; 
    entry_t* evt = this -> add_entry(hash); 
    return evt -> has_selection(sel);
}

void container::compile(size_t* l){
    std::map<std::string, entry_t*>::iterator itr = this -> random_access -> begin();  
    for (; itr != this -> random_access -> end(); ++itr){
        entry_t* ev = itr -> second;  
        for (event_template* evx : *ev -> m_event){evx -> CompileEvent();}
        if (ev -> m_selection -> size() && !this -> merged){
            this -> merged = new std::map<std::string, selection_template*>();
        }

        for (selection_template* sel : *ev -> m_selection){
            sel -> CompileEvent(); 
            sel -> m_event = nullptr; 
            std::string name = sel -> name; 

            if (!this -> merged -> count(name)){(*this -> merged)[name] = sel -> clone();}
            (*this -> merged)[name] -> merger(sel); 
        }

        for (graph_template* gr : *ev -> m_graph){
            gr -> CompileEvent(); 
            gr -> flush_particles();
            graph_t* gr_ = gr -> data_export();  
            gr_ -> hash = ev -> hash;
            gr_ -> filename = this -> filename; 
            ev -> m_data -> push_back(gr_); 
        }
        ev -> destroy(); 
        *l += 1; 
    }
    *l = this -> random_access -> size(); 
}

void container::fill_selections(std::map<std::string, selection_template*>* inpt){
    if (!this -> merged){return;}
    std::map<std::string, selection_template*>::iterator itr;
    itr = this -> merged -> begin();
    for (; itr != this -> merged -> end(); ++itr){
        selection_template* sl = (*inpt)[itr -> first]; 
        sl -> merger(itr -> second); 
        delete itr -> second; 
    }
    this -> merged -> clear(); 
    delete this -> merged;
    this -> merged = nullptr; 
}

void container::populate_dataloader(dataloader* dl){
    std::map<std::string, entry_t*>::iterator itr = this -> random_access -> begin();  
    for (; itr != this -> random_access -> end(); ++itr){
        entry_t* ev = itr -> second; 
        for (graph_t* gr_ : *ev -> m_data){dl -> extract_data(gr_);}
        ev -> m_data -> clear(); 
        delete ev -> m_data; 
    }
}

size_t container::len(){return this -> random_access -> size();}
