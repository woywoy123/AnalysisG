#include <container/container.h>

container::container(){}
container::~container(){
    delete this -> meta_data; 
    delete this -> filename;  

    std::map<std::string, entry_t>::iterator itr = this -> random_access.begin();
    for (; itr != this -> random_access.end(); ++itr){itr -> second.destroy();}
    this -> random_access.clear(); 
    if (!this -> merged){return;}
    std::map<std::string, selection_template*>::iterator itrs = this -> merged -> begin(); 
    for (; itrs != this -> merged -> end(); ++itrs){delete itrs -> second; itrs -> second = nullptr;}
    this -> merged -> clear(); 
    delete this -> merged; 
    this -> merged = nullptr; 
}

void container::get_events(std::vector<event_template*>* out, std::string label){
    if (label != this -> label && label.size()){return;}
    std::map<std::string, entry_t>::iterator itr = this -> random_access.begin(); 
    for (; itr != this -> random_access.end(); ++itr){
        std::vector<event_template*> rn = itr -> second.m_event; 
        out -> insert(out -> end(), rn.begin(), rn.end()); 
    } 
}

void container::add_meta_data(meta* data, std::string fname){
    this -> filename = new std::string(fname); 
    this -> meta_data = data; 
}

meta* container::get_meta_data(){return this -> meta_data;}

entry_t* container::add_entry(std::string hash){
    if (this -> random_access.count(hash)){return &this -> random_access[hash];}
    entry_t* t = &this -> random_access[hash]; 
    t -> init(); 
    t -> hash = hash;  
    return t; 
}

bool container::add_event_template(event_template* ev, std::string _label){
    if (!this -> label.size()){this -> label = _label;}
    entry_t* evt = this -> add_entry(ev -> hash); 
    ev -> meta_data = this -> meta_data; 
    return evt -> has_event(ev); 
}

bool container::add_graph_template(graph_template* gr, std::string _label){
    if (!this -> label.size()){this -> label = _label;}
    entry_t* evt = this -> add_entry(gr -> hash); 
    gr -> meta_data = this -> meta_data; 
    return evt -> has_graph(gr); 
}

bool container::add_selection_template(selection_template* sel){
    entry_t* evt = this -> add_entry(sel -> hash); 
    sel -> meta_data = this -> meta_data; 
    return evt -> has_selection(sel);
}

void container::compile(size_t* l){
    std::map<std::string, entry_t>::iterator itr = this -> random_access.begin();  
    for (; itr != this -> random_access.end(); ++itr){
        entry_t* ev = &itr -> second;  
        for (event_template* evx : ev -> m_event){evx -> CompileEvent();}
        if (ev -> m_selection.size() && !this -> merged){
            this -> merged = new std::map<std::string, selection_template*>();
        }

        for (selection_template* sel : ev -> m_selection){
            std::string name = sel -> name; 
            if (!this -> merged -> count(name)){(*this -> merged)[name] = sel -> clone();}
            bool passed = sel -> CompileEvent(); 
            if (passed){(*this -> merged)[name] -> merger(sel);}
            sel -> m_event = nullptr; 
        }

        for (graph_template* gr : ev -> m_graph){
            if (!gr -> PreSelection()){continue;}
            gr -> CompileEvent(); 
            gr -> flush_particles();
            graph_t* gr_    = gr -> data_export();  
            gr_ -> hash     = new std::string(ev -> hash);
            gr_ -> filename = this -> filename; 
            ev -> m_data.push_back(gr_); 
        }
        ev -> destroy(); 
        *l += 1; 
    }
    *l = this -> random_access.size(); 
}

void container::fill_selections(std::map<std::string, selection_template*>* inpt){
    if (!this -> merged){return;}
    std::map<std::string, selection_template*>::iterator itr = this -> merged -> begin();
    for (; itr != this -> merged -> end(); ++itr){
        selection_template* sl = (*inpt)[itr -> first]; 
        sl -> merger(itr -> second); 
        delete itr -> second; 
        itr -> second = nullptr; 
    }
    this -> merged -> clear(); 
    delete this -> merged;
    this -> merged = nullptr; 
}

void container::populate_dataloader(dataloader* dl){
    std::map<std::string, entry_t>::iterator itr = this -> random_access.begin();  
    for (; itr != this -> random_access.end(); ++itr){
        std::vector<graph_t*> data = itr -> second.m_data; 
        for (size_t x(0); x < data.size(); ++x){dl -> extract_data(data[x]);}
        itr -> second.m_data.clear(); 
    }
    this -> random_access.clear(); 
}

size_t container::len(){return this -> random_access.size();}
