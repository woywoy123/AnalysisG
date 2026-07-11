#include <generators/sampletracer.h>

sampletracer::sampletracer(){}
sampletracer::~sampletracer(){}

bool sampletracer::add_meta_data(meta* meta_, std::string filename){
    if (this -> root_container.count(filename)){return false;}
    this -> root_container[filename].add_meta_data(meta_, filename); 
    return true; 
}

meta* sampletracer::get_meta_data(std::string filename){
    if (!this -> root_container.count(filename)){return nullptr;}
    return this -> root_container[filename].get_meta_data(); 
}

std::vector<event_template*> sampletracer::get_events(std::string label){
    std::vector<event_template*> out = {};
    std::map<std::string, container>::iterator itr = this -> root_container.begin(); 
    for (; itr != this -> root_container.end(); ++itr){itr -> second.get_events(&out, label);}
    return out; 
}

bool sampletracer::add_event(event_template* ev, std::string label){
    return this -> root_container[ev -> filename].add_event_template(ev, label); 
}

bool sampletracer::add_graph(graph_template* gr, std::string label){
    return this -> root_container[gr -> filename].add_graph_template(gr, label); 
}

bool sampletracer::add_selection(selection_template* sel){
    return this -> root_container[sel -> filename].add_selection_template(sel); 
}

void sampletracer::compile_objects(int threads, int intrath){
    auto lamb = [](tracing_t* tr, container* data, int intrath){
        data -> compile(tr -> idx, tr -> threadIdx, intrath);
        tr -> finished(); 
    }; 

    multithreaded_t* thr = this -> make_threads(this -> root_container.size(), threads); 
    std::map<std::string, container>::iterator itr = this -> root_container.begin(); 
    for (; itr != this -> root_container.end(); ++itr){
        tracing_t* tr = thr -> next(); 
        tr -> register_thread( new std::thread(lamb, tr, &itr -> second, intrath), itr -> second.len() ); 
        this -> await_threads(thr, false); 
    }
    this -> await_threads(thr, true); 
    this -> pflush(&thr); 
}

void sampletracer::populate_dataloader(dataloader* dl){
    std::map<std::string, container>::iterator itr = this -> root_container.begin(); 
    for (; itr != this -> root_container.end(); ++itr){itr -> second.populate_dataloader(dl);}
}

bool sampletracer::fill_selections(std::map<std::string, selection_template*>* inpt){
    if (!inpt -> size()){return false;}
    std::map<std::string, container>::iterator itr = this -> root_container.begin(); 
    for (; itr != this -> root_container.end(); ++itr){itr -> second.fill_selections(inpt);}
    return true;
}

