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

    for (entry_t* ev : *this -> random_access){
        ev -> destroy(); 
        delete ev -> hash; 
        delete ev; 
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
    for (entry_t* ev : *this -> random_access){
        std::vector<event_template*>* rn = ev -> m_event; 
        out -> insert(out -> end(), rn -> begin(), rn -> end()); 
    } 
}

void container::add_meta_data(meta* data, std::string fname){
    this -> filename = new std::string(fname); 
    this -> meta_data = data; 
}

meta* container::get_meta_data(){return this -> meta_data;}

bool container::add_event_template(event_template* ev, std::string _label, long* alloc){
    if (!this -> label){this -> label = new std::string(_label);}
    if (!this -> alloc){
        this -> random_access -> reserve(*alloc);
        this -> alloc = *alloc; 
    }

    std::string hash = ev -> hash; 
    if (!this -> hash_map -> count(hash)){
        entry_t* t = new entry_t();
        t -> init(); 
        t -> hash = new std::string(hash); 
        t -> meta_data = this -> meta_data; 

        (*this -> hash_map)[hash] = this -> hash_map -> size();
        this -> random_access -> push_back(t); 
    }

    return (*this -> random_access)[(*this -> hash_map)[hash]] -> has_event(ev); 
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
    }
    return (*this -> random_access)[(*this -> hash_map)[hash]] -> has_graph(gr); 
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
    }
    return (*this -> random_access)[(*this -> hash_map)[hash]] -> has_selection(sel);
}


void container::compile(){
    for (int x(0); x < this -> random_access -> size(); ++x){
        entry_t* ev = this -> random_access -> at(x); 
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
        entry_t* ev = (*this -> random_access)[x]; 
        for (graph_t* gr_ : *ev -> m_data){dl -> extract_data(gr_);}
    }
}

