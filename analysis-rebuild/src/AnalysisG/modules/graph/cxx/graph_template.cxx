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

void graph_template::flush_particles(){
    this -> m_event = nullptr; 
    this -> _topology = {}; 
    this -> _topological_index = {}; 
    this -> nodes = {}; 
    this -> node_particles = {}; 
}


void graph_template::define_particle_nodes(std::vector<particle_template*>* prt){
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

    this -> _topology = {}; 
    this -> _topological_index = {}; 

    int idx = 0; 
    std::vector<int> src_, dst_ = {}; 
    for (itr1 = this -> node_particles.begin(); itr1 != this -> node_particles.end(); ++itr1){
        for (itr2 = this -> node_particles.begin(); itr2 != this -> node_particles.end(); ++itr2){
            if (!fx(itr1 -> second, itr2 -> second)){
                this -> _topology.push_back({}); 
                this -> _topological_index.push_back(-1); 
                continue;
            }

            src_.push_back(itr1 -> first); 
            dst_.push_back(itr2 -> first);  
            this -> _topology.push_back({itr1 -> first, itr2 -> first}); 
            this -> _topological_index.push_back(idx); 
            ++idx; 
        }
    }
    torch::Tensor t1 = this -> to_tensor(src_, torch::kInt).view({1, -1}); 
    torch::Tensor t2 = this -> to_tensor(dst_, torch::kInt).view({1, -1}); 
    this -> m_topology = torch::cat({t1, t2}, 0); 
}

graph_t* graph_template::data_export(){
    
    std::map<std::string, torch::Tensor*> g_tru_t = {}; 
    std::map<std::string, torch::Tensor*> g_dat_t = {};

    std::map<std::string, int>* g_tru_i = new std::map<std::string, int>();
    std::map<std::string, int>* g_dat_i = new std::map<std::string, int>();
    this -> build_export(&g_tru_t, g_tru_i, &g_dat_t, g_dat_i, &this -> graph_fx); 
    
    std::map<std::string, torch::Tensor*> n_tru_t = {}; 
    std::map<std::string, torch::Tensor*> n_dat_t = {};

    std::map<std::string, int>* n_tru_i = new std::map<std::string, int>();
    std::map<std::string, int>* n_dat_i = new std::map<std::string, int>();
    this -> build_export(&n_tru_t, n_tru_i, &n_dat_t, n_dat_i, &this -> node_fx); 
    
    std::map<std::string, torch::Tensor*> e_tru_t = {}; 
    std::map<std::string, torch::Tensor*> e_dat_t = {};

    std::map<std::string, int>* e_tru_i = new std::map<std::string, int>();
    std::map<std::string, int>* e_dat_i = new std::map<std::string, int>();
    this -> build_export(&e_tru_t, e_tru_i, &e_dat_t, e_dat_i, &this -> edge_fx); 
    
    graph_t* gr = new graph_t(); 
    gr -> edge_index = new torch::Tensor(this -> m_topology);  
    gr -> add_truth_graph(&g_tru_t, g_tru_i); 
    gr -> add_truth_node(&n_tru_t, n_tru_i); 
    gr -> add_truth_edge(&e_tru_t, e_tru_i); 

    gr -> add_data_graph(&g_dat_t, g_dat_i); 
    gr -> add_data_node(&n_dat_t, n_dat_i); 
    gr -> add_data_edge(&e_dat_t, e_dat_i); 
    
    gr -> num_nodes   = this -> num_nodes; 
    gr -> event_index = this -> index; 
    return gr; 
}

graph_template* graph_template::build_event(event_template* ev){
    event_t* data_ = &ev -> data; 
    data_ -> name = this -> name; 

    graph_template* gr = this -> clone(); 
    gr -> m_event = ev; 
    gr -> data = *data_; 
    gr -> filename = ev -> filename; 
    return gr; 
}

void graph_template::CompileEvent(){}
