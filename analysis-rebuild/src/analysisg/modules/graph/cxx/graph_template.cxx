#include <templates/graph_template.h>

graph_template::graph_template(){
    this -> name.set_getter(this -> set_name); 
    this -> name.set_object(this); 

    this -> hash.set_getter(this -> get_hash); 
    this -> hash.set_object(this); 

    this -> tree.set_getter(this -> get_tree); 
    this -> tree.set_object(this); 

    this -> index.set_getter(this -> get_index); 
    this -> index.set_object(this);

    this -> device.set_setter(this -> set_device); 
    this -> device.set_getter(this -> get_device); 
    this -> device.set_object(this); 
}

graph_template::~graph_template(){
    if (!this -> op){return;}
    delete this -> op;
}

bool graph_template::operator == (graph_template& p){
    return this -> hash == p.hash; 
}

graph_template* graph_template::clone(){
    return new graph_template(); 
}

void graph_template::add_particle_nodes(std::vector<particle_template*>* prt){
    for (size_t x(0); x < prt -> size(); ++x){
        std::string hash_ = prt -> at(x) -> hash; 
        if (this -> nodes.count(hash_)){continue;}

        int n_nodes = (int)this -> nodes.size();
        this -> node_particles[n_nodes] = prt -> at(x);  
        this -> nodes[hash_] = n_nodes; 
    } 
}

void graph_template::define_topology(std::function<bool(particle_template*, particle_template*)> fx){
    std::map<int, particle_template*>::iterator itr1;
    std::map<int, particle_template*>::iterator itr2;

    std::vector<int> src_, dst_ = {}; 
    for (itr1 = this -> node_particles.begin(); itr1 != this -> node_particles.end(); ++itr1){
        for (itr2 = this -> node_particles.begin(); itr2 != this -> node_particles.end(); ++itr2){
            if (!fx(itr1 -> second, itr2 -> second)){continue;}
            src_.push_back(itr1 -> first); 
            dst_.push_back(itr2 -> first);  
        }
    }
    torch::Tensor t1 = this -> to_tensor(src_, torch::kInt).view({1, -1}); 
    torch::Tensor t2 = this -> to_tensor(dst_, torch::kInt).view({1, -1}); 
    this -> m_topology = torch::cat({t1, t2}, 0); 
}

void graph_template::build_event(event_template* ev){}
void graph_template::CompileEvent(){}
