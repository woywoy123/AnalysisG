#include <generators/sampletracer.h>

sampletracer::sampletracer(){
    this -> root_container = new std::map<std::string, container*>(); 
}

sampletracer::~sampletracer(){
    std::map<std::string, container*>::iterator itr;
    itr = this -> root_container -> begin(); 
    for (; itr != this -> root_container -> end(); ++itr){delete itr -> second;}
    delete this -> root_container; 
}

bool sampletracer::add_meta_data(meta* meta_, std::string filename){
    if (this -> root_container -> count(filename)){return false;}
    container* con = new container();
    con -> add_meta_data(meta_, filename); 
    (*this -> root_container)[filename] = con; 
    return true; 
}

meta* sampletracer::get_meta_data(std::string filename){
    if (!this -> root_container -> count(filename)){return nullptr;}
    return (*this -> root_container)[filename] -> get_meta_data(); 
}

std::vector<event_template*>* sampletracer::get_events(std::string label){
    std::vector<event_template*>* out = new std::vector<event_template*>();
    std::map<std::string, container*>::iterator itr;
    itr = this -> root_container -> begin(); 
    for (; itr != this -> root_container -> end(); ++itr){
        itr -> second -> get_events(out, label);
    }
    out -> shrink_to_fit();
    return out; 
}

bool sampletracer::add_event(event_template* ev, std::string label){
    std::string fname = ev -> filename; 
    container* con = (*this -> root_container)[fname]; 
    return con -> add_event_template(ev, label); 
}

bool sampletracer::add_graph(graph_template* gr, std::string label){
    std::string fname = gr -> filename; 
    container* con = (*this -> root_container)[fname]; 
    return con -> add_graph_template(gr, label); 
}

bool sampletracer::add_selection(selection_template* sel){
    std::string fname = sel -> filename; 
    container* con = (*this -> root_container)[fname]; 
    return con -> add_selection_template(sel); 
}

void sampletracer::compile_objects(){
    auto lamb = [](container* data){data -> compile();}; 

    int index = 0; 
    std::vector<std::thread*> threads_(this -> root_container -> size(), nullptr); 
    std::map<std::string, container*>::iterator itr = this -> root_container -> begin(); 
    for (; itr != this -> root_container -> end(); ++itr, ++index){
        threads_[index] = new std::thread(lamb, itr -> second); 
    }
    for (int x(0); x < index; ++x){
        threads_[x] -> join();
        delete threads_[x]; 
    }  
}

void sampletracer::populate_dataloader(dataloader* dl){
    std::map<std::string, container*>::iterator itr = this -> root_container -> begin(); 
    for (; itr != this -> root_container -> end(); ++itr){
        itr -> second -> populate_dataloader(dl);
    }
}

void sampletracer::fill_selections(std::map<std::string, selection_template*>* inpt){
    std::map<std::string, container*>::iterator itr = this -> root_container -> begin(); 
    for (; itr != this -> root_container -> end(); ++itr){
        itr -> second -> fill_selections(inpt);
    }
}

