#include <container/container.h>

container::container(){
    this -> random_access = new std::vector<entry_t*>(); 
    this -> hash_map      = new std::map<std::string, int>(); 
}

container::~container(){
    delete this -> hash_map; 
    delete this -> meta_data; 
    delete this -> filename;  
    delete this -> label; 

    for (size_t x(0); x < this -> random_access -> size(); ++x){
        this -> random_access -> at(x) -> destroy(); 
        delete this -> random_access -> at(x); 
    }
    delete this -> random_access; 

    if (!this -> merged){return;}
    std::map<std::string, selection_template*>::iterator itr;
    itr = this -> merged -> begin(); 
    for (; itr != this -> merged -> end(); ++itr){delete itr -> second;}
    delete this -> merged; 
}

void container::get_events(std::vector<event_template*>* out, std::string label){
    if (label != *this -> label && label.size()){return;}
    for (size_t x(0); x < this -> random_access -> size(); ++x){
        std::vector<event_template*>* rn = nullptr; 
        rn = this -> random_access -> at(x) -> m_event; 
        out -> insert(out -> end(), rn -> begin(), rn -> end()); 
    } 
}

void container::add_meta_data(meta* data, std::string fname){
    this -> filename = new std::string(fname); 
    this -> meta_data = data; 
}

bool container::add_event_template(event_template* ev, std::string _label){
    if (!this -> label){this -> label = new std::string(_label);}
    std::string hash = ev -> hash; 
    if (!this -> hash_map -> count(hash)){
        entry_t* t = new entry_t();
        t -> init(); 
        t -> hash = new std::string(hash); 
        t -> meta_data = this -> meta_data; 

        (*this -> hash_map)[hash] = this -> hash_map -> size();
        this -> random_access -> push_back(t); 
        this -> random_access -> shrink_to_fit(); 
    }
    entry_t* en = (*this -> random_access)[(*this -> hash_map)[hash]];
    return en -> has_event(ev); 
}

bool container::add_graph_template(graph_template* gr, std::string _label){
    if (!this -> label){this -> label = new std::string(_label);}

    std::string hash = gr -> hash; 
    if (!this -> hash_map -> count(hash)){
        entry_t* t = new entry_t();
        t -> init(); 
        t -> hash = new std::string(hash); 
        t -> meta_data = this -> meta_data; 

        (*this -> hash_map)[hash] = this -> hash_map -> size();
        this -> random_access -> push_back(t); 
        this -> random_access -> shrink_to_fit(); 
    }
    entry_t* en = (*this -> random_access)[(*this -> hash_map)[hash]];
    return en -> has_graph(gr); 
}

bool container::add_selection_template(selection_template* sel){
    std::string hash = sel -> hash; 
    if (!this -> hash_map -> count(hash)){
        entry_t* t = new entry_t();
        t -> init(); 
        t -> hash = new std::string(hash); 
        t -> meta_data = this -> meta_data; 

        (*this -> hash_map)[hash] = this -> hash_map -> size();
        this -> random_access -> push_back(t); 
        this -> random_access -> shrink_to_fit(); 
    }
    entry_t* en = (*this -> random_access)[(*this -> hash_map)[hash]];
    return en -> has_selection(sel); 
}


void container::compile(){
    for (int x(0); x < this -> random_access -> size(); ++x){
        entry_t* ev = this -> random_access -> at(x); 
        for (size_t y(0); y < ev -> m_event -> size(); ++y){
            ev -> m_event -> at(y) -> CompileEvent(); 
        }

        if (ev -> m_selection -> size() && !this -> merged){
            this -> merged = new std::map<std::string, selection_template*>();
        }

        for (size_t y(0); y < ev -> m_selection -> size(); ++y){
            selection_template* sel = ev -> m_selection -> at(y); 
            sel -> CompileEvent(); 
            sel -> m_event = nullptr; 
            std::string name = sel -> name; 

            if (!this -> merged -> count(name)){(*this -> merged)[name] = sel -> clone();}
            (*this -> merged)[name] -> merger(sel); 
        }

        for (size_t y(0); y < ev -> m_graph -> size(); ++y){
            graph_template* gr = ev -> m_graph -> at(y); 
            gr -> CompileEvent(); 
            gr -> flush_particles();
            graph_t* gr_ = gr -> data_export();  
            gr_ -> hash = ev -> hash;
            ev -> m_data -> push_back(gr_); 
        }
        
        ev -> destroy(ev -> m_event); 
        ev -> destroy(ev -> m_graph);
        ev -> destroy(ev -> m_selection); 
        ev -> m_data -> shrink_to_fit(); 
    }
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
    delete this -> merged;
    this -> merged = nullptr; 
}

void container::populate_dataloader(dataloader* dl){
    for (int x(0); x < this -> random_access -> size(); ++x){
        entry_t* ev = this -> random_access -> at(x); 
        for (size_t t(0); t < ev -> m_data -> size(); ++t){
            dl -> extract_data(ev -> m_data -> at(t));
        }
    }
}

