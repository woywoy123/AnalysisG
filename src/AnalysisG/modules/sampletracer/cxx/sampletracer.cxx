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

void sampletracer::compile_objects(int threads){
    auto lamb = [](size_t* l, container* data){data -> compile(l);}; 

    int index = 0; 
    std::vector<size_t> progres(this -> root_container.size(), 0); 
    std::vector<size_t> handles(this -> root_container.size(), 0); 
    std::vector<std::string*> titles_(this -> root_container.size(), nullptr); 
    std::vector<std::thread*> threads_(this -> root_container.size(), nullptr); 

    std::map<std::string, container>::iterator itr = this -> root_container.begin(); 
    for (int x(0); itr != this -> root_container.end(); ++itr, ++x){
        progres[x] = itr -> second.len();
        std::vector<std::string> vec = this -> split(itr -> first, "/"); 
        titles_[x] = new std::string(vec[vec.size()-1]); 
    }
    if (!this -> tools::sum(&progres)){return;}
    std::thread* thr = new std::thread(this -> progressbar3, &handles, &progres, &titles_);

    int tidx = 0; 
    itr = this -> root_container.begin(); 
    for (; itr != this -> root_container.end(); ++itr, ++index, ++tidx){
        threads_[index] = new std::thread(lamb, &handles[index], &itr -> second); 
        while (tidx > threads-1){tidx = this -> running(&threads_);}
    }
    this -> monitor(&threads_); 
    thr -> join(); delete thr; thr = nullptr; 
}

void sampletracer::populate_dataloader(dataloader* dl){
    std::map<std::string, container>::iterator itr = this -> root_container.begin(); 
    for (; itr != this -> root_container.end(); ++itr){itr -> second.populate_dataloader(dl);}
}

void sampletracer::fill_selections(std::map<std::string, selection_template*>* inpt){
    std::map<std::string, container>::iterator itr = this -> root_container.begin(); 
    for (; itr != this -> root_container.end(); ++itr){itr -> second.fill_selections(inpt);}
}

