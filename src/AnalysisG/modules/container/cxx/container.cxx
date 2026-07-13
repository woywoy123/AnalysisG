#include <container/container.h>
#include <tools/merge_cast.h>
#include <TSystem.h>

container::container(){
    this -> merged = new std::map<std::string, selection_template*>();
}

container::~container(){
    this -> pflush(&this -> meta_data); 
    this -> pflush(&this -> filename); 
    this -> random_access.clear(); 
    this -> mflush(this -> merged); 
    this -> pflush(&this -> merged); 
}

void container::get_events(std::vector<event_template*>* out, std::string _label){
    if (_label != this -> label && _label.size()){return;}
    std::map<std::string, entry_t>::iterator itr = this -> random_access.begin(); 
    for (; itr != this -> random_access.end(); ++itr){ merge_data(out, &itr -> second.m_event); } 
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
    entry_t*    evt = this -> add_entry(ev -> hash); 
    ev -> meta_data = this -> meta_data; 
    return evt -> has_event(ev); 
}

bool container::add_graph_template(graph_template* gr, std::string _label){
    if (!this -> label.size()){this -> label = _label;}
    entry_t*    evt = this -> add_entry(gr -> hash); 
    gr -> meta_data = this -> meta_data; 
    return evt -> has_graph(gr); 
}

bool container::add_selection_template(selection_template* sel){
    entry_t*     evt = this -> add_entry(sel -> hash); 
    sel -> meta_data = this -> meta_data; 
    return evt -> has_selection(sel);
}

void container::compile(size_t* l, int threadIdx, int thrds){
    auto lmb =[this](
        std::vector<event_template*>* evc, 
        std::vector<graph_template*>* grc,
        tracing_t* lx
    ) -> void{
        lx -> message(tools::get_splits(this -> filename, "/")); 
        for (size_t x(0); x < evc -> size(); ++x){
            (*evc)[x] -> CompileEvent(); 
            lx -> next(); 
        }

        for (size_t x(0); x < grc -> size(); ++x){
            graph_template* gr = (*grc)[x]; 
            lx -> next(); 
            if (gr -> preselection && !gr -> PreSelection()){}
            else {gr -> CompileEvent();}
            //gr -> flush_particles();
        }
        lx -> finished(); 
    }; 

    std::vector<event_template*> ev_vec;
    std::vector<graph_template*> gr_vec;

    std::map<std::string, entry_t>::iterator itr = this -> random_access.begin(); 
    for (; itr != this -> random_access.end(); ++itr){
        merge_data(&ev_vec, &itr -> second.m_event); 
        merge_data(&gr_vec, &itr -> second.m_graph);  
        for (graph_template* gr : itr -> second.m_graph){gr -> threadIdx = threadIdx;}
    }
 
    if (thrds < 0){thrds = 2;} 
    std::vector<std::vector<event_template*>> ev_vx = this -> discretize(&ev_vec, ev_vec.size() / thrds ); 
    std::vector<std::vector<graph_template*>> gr_vx = this -> discretize(&gr_vec, gr_vec.size() / thrds ); 
    size_t itz = ev_vx.size();  
    if (itz == 0){return;}

//    this -> shush = true; 
    multithreaded_t* thr = this -> make_threads(itz, thrds); 
    for (size_t x(0); x < itz; ++x){
        tracing_t* tr = thr -> next(); 
        tr -> register_thread( new std::thread(lmb, &ev_vx[x], &gr_vx[x], tr), ev_vx[x].size() + gr_vx[x].size() ); 
//        while (this -> await_threads(thr, true)){}
    }
    while (this -> await_threads(thr, true)){}
    this -> pflush(&thr); 
    std::map<std::string, write_t*> handles;

    itr = this -> random_access.begin(); 
    for (; itr != this -> random_access.end(); ++itr){
        entry_t* ev = &itr -> second;  
        for (graph_template* gr : ev -> m_graph){
            graph_t* gr_    = gr -> data_export();  
            gr_ -> hash     = new std::string(ev -> hash);
            gr_ -> filename = this -> filename; 
            ev -> m_data.push_back(gr_); 
        }

        if (itr == this -> random_access.begin()){
            for (selection_template* sel : ev -> m_selection){
                if (!this -> output_path){break;}
                std::string name = sel -> name; 
                std::string tree = sel -> m_event -> tree; 
                std::string pth  = *this -> output_path + "/Selections/"; 
                pth += (name + "-" + std::string(sel -> m_event -> name) + "/");

                if (this -> label.size()){pth += this -> label + "/";}
                this -> create_path(pth); 
                std::string fname = this -> get_splits(this -> filename, "/"); 

                handles[name] = new write_t(); 
                handles[name] -> mtx = &this -> meta_data -> meta_data; 
                handles[name] -> create(tree, pth + fname); 
            }
            for (selection_template* sel : ev -> m_selection){
                std::string name = sel -> name; 
                selection_template* slx = sel -> clone(); 

                slx -> threadIdx = threadIdx; 
                if (this -> output_path){slx -> handle = handles[name];}
                (*this -> merged)[name] = slx;
            }
        }

        for (selection_template* sel : ev -> m_selection){
            std::string name = sel -> name; 
            sel -> threadIdx = threadIdx; 
            sel -> bulk_write(nullptr, nullptr); 
            if (this -> output_path){sel -> handle = handles[name];}
            bool col = sel -> CompileEvent();
            if (col){(*this -> merged)[name] -> merger(sel);}
            if (this -> output_path && !sel -> p_bulk_write){handles[name] -> write();}
            sel -> m_event = nullptr; 
        }
        (*l) += 1; 
        ev -> destroy(); 
    }

    std::map<std::string, write_t*>::iterator itx = handles.begin(); 
    for (; itx != handles.end(); ++itx){
        (*this -> merged)[itx -> first] -> bulk_write_out();
        (*this -> merged)[itx -> first] -> handle = nullptr; 
        itx -> second -> close();
        this -> pflush(&itx -> second); 
    }
}

void container::fill_selections(std::map<std::string, selection_template*>* inpt){
    if (!this -> merged){return;}
    std::map<std::string, selection_template*>::iterator itr = this -> merged -> begin();
    for (; itr != this -> merged -> end(); ++itr){
        selection_template* sl = (*inpt)[itr -> first]; 
        sl -> merger(itr -> second); 
        this -> pflush(&itr -> second); 
    }
    this -> pflush(&this -> merged); 
}

void container::populate_dataloader(dataloader* dl){
    std::map<std::string, entry_t>::iterator itr = this -> random_access.begin();  
    std::cout << "________________" << std::endl; 
    for (; itr != this -> random_access.end(); ++itr){
        std::vector<graph_t*> data = itr -> second.m_data; 
        std::cout << "________________" << std::endl; 
//        for (size_t x(0); x < data.size(); ++x){dl -> extract_data(data[x]);}
        itr -> second.m_data.clear(); 
    }
    std::cout << this -> random_access.size() << std::endl; 
    this -> random_access.clear(); 
    abort(); 
}

size_t container::len(){return this -> random_access.size();}
