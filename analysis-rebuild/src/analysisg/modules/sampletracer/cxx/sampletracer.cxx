#include <generators/sampletracer.h>

container::container(){
    this -> event_tree = new std::vector<std::map<std::string, event_template*>>(); 
    this -> graph_tree = new std::vector<std::map<std::string, graph_template*>>(); 
}

container::~container(){
    for (size_t x(0); x < this -> event_tree -> size(); ++x){
        std::map<std::string, event_template*>::iterator ite = (*this -> event_tree)[x].begin(); 
        for (; ite != (*this -> event_tree)[x].end(); ++ite){delete ite -> second;}
    }
    delete this -> event_tree; 

    for (size_t x(0); x < this -> graph_tree -> size(); ++x){
        std::map<std::string, graph_template*>::iterator ite = (*this -> graph_tree)[x].begin(); 
        for (; ite != (*this -> graph_tree)[x].end(); ++ite){delete ite -> second;}
    }
    delete this -> graph_tree; 
}

void container::register_event(std::map<std::string, event_template*>* inpt){
    register_object(inpt, this -> event_tree, &this -> event_hash_map); 
}

void container::register_event(std::map<std::string, graph_template*>* inpt){
    register_object(inpt, this -> graph_tree, &this -> graph_hash_map); 
}

std::vector<event_template*> container::get_event(std::string name, std::string tree){
    std::vector<event_template*> output; 
    find_object(&output, this -> event_tree, name, tree); 
    return output; 
}

std::vector<graph_template*> container::get_graph(std::string name, std::string tree){
    std::vector<graph_template*> output; 
    find_object(&output, this -> graph_tree, name, tree); 
    return output; 
}

bool container::threaded_compilation(){
    //std::vector<event_template*> events = {}; 
    //std::map<std::string, int>::iterator itr = this -> hash_map.begin();
    //for (; itr != hash_map.end(); ++itr){
    //    std::map<std::string, event_template*>::iterator ite; 
    //    ite = this -> event_tree[itr -> second].begin();
    //    for (; ite != this -> event_tree[itr -> second].end(); ++ite){
    //        if (this -> compiled[itr -> first][ite -> first]){continue;}
    //        this -> compiled[itr -> first][ite -> first] = true; 
    //        events.push_back(ite -> second); 
    //    }
    //}

    //std::vector<std::vector<event_template*>> quant;
    //quant = tools().discretize(&events, (int)(events.size()/this -> threads)); 

    //bool execute = true;
    //std::vector<std::thread*> th; 
    //for (int x(0); x < quant.size(); ++x){
    //    th.push_back(new std::thread(process_data<event_template>, &quant[x], &execute)); 
    //}

    //execute = false;  
    //for (int x(0); x < this -> threads; ++x){th[x] -> join(); delete th[x];}
    return true; 
}


sampletracer::sampletracer(){
    this -> prefix = "SampleTracer";
    this -> root_container = new std::map<std::string, container*>(); 
}

std::vector<event_template*> sampletracer::get_event(std::string type, std::string tree){
    std::vector<event_template*> output = {}; 
    std::map<std::string, container*>::iterator itr = this -> root_container -> begin(); 
    for (; itr != this -> root_container -> end(); ++itr){
        std::vector<event_template*> x = itr -> second -> get_event(type, tree); 
        output.insert(output.end(), x.begin(), x.end()); 
    }
    return output; 
}

std::vector<graph_template*> sampletracer::get_graph(std::string type, std::string tree){
    std::vector<graph_template*> output = {}; 
    std::map<std::string, container*>::iterator itr = this -> root_container -> begin(); 
    for (; itr != this -> root_container -> end(); ++itr){
        std::vector<graph_template*> x = itr -> second -> get_graph(type, tree); 
        output.insert(output.end(), x.begin(), x.end()); 
    }
    return output; 
}

sampletracer::~sampletracer(){
    std::map<std::string, container*>::iterator itr;
    itr = this -> root_container -> begin(); 
    for (; itr != this -> root_container -> end(); ++itr){delete itr -> second;}
    delete this -> root_container; 
}



