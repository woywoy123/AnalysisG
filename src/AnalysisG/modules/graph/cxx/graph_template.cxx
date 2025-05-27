#include <templates/graph_template.h>

graph_template::graph_template(){
    this -> op = new torch::TensorOptions(torch::kCPU);
    this -> name.set_setter(this -> set_name); 
    this -> name.set_object(this); 

    this -> preselection.set_setter(this -> set_preselection); 
    this -> preselection.set_getter(this -> get_preselection); 
    this -> preselection.set_object(this); 

    this -> hash.set_getter(this -> get_hash); 
    this -> hash.set_object(this); 

    this -> tree.set_getter(this -> get_tree); 
    this -> tree.set_object(this); 

    this -> index.set_getter(this -> get_index); 
    this -> index.set_object(this);

    this -> weight.set_getter(this -> get_weight); 
    this -> weight.set_object(this);
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
    for (size_t x(0); x < this -> garbage.size(); ++x){
        if (!this -> garbage[x]){continue;}
        delete this -> garbage[x];
    }
    this -> garbage.clear();  
}

void graph_template::define_particle_nodes(std::vector<particle_template*>* prt){
    for (size_t x(0); x < prt -> size(); ++x){
        std::string hash_ = (*prt)[x] -> hash; 
        if (this -> nodes.count(hash_)){continue;}
        int n_nodes = (int)this -> nodes.size();
        this -> node_particles[n_nodes] = (*prt)[x]; 
        this -> nodes[hash_] = n_nodes; 
    }
    this -> num_nodes = this -> nodes.size(); 
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
    torch::Tensor t1 = this -> to_tensor(src_, torch::kInt, int()).view({1, -1}); 
    torch::Tensor t2 = this -> to_tensor(dst_, torch::kInt, int()).view({1, -1}); 
    this -> m_topology = torch::cat({t1, t2}, 0); 
}

std::pair<particle_template*, particle_template*> graph_template::double_neutrino(
        std::vector<particle_template*> particles, double met, double phi, std::string device, 
        double top_mass, double wboson_mass, double distance, double perturb, long steps
){
    if (!particles.size()){return {nullptr, nullptr};}
    std::vector<std::pair<neutrino*, neutrino*>> nux; 
    std::vector<double> metv = std::vector<double>({met}); 
    std::vector<double> phiv = std::vector<double>({phi});
    std::vector<std::vector<particle_template*>> prt = {particles}; 

    #ifdef PYC_CUDA
    nux = pyc::nusol::combinatorial(metv, phiv, prt, device, top_mass, wboson_mass, distance, perturb, steps); 
    #endif

    if (!nux.size()){return {nullptr, nullptr};}
    double val1(0), val2(0); 
    particle_template* n1 = nullptr; 
    particle_template* n2 = nullptr; 
    for (size_t x(0); x < nux.size(); ++x){
        neutrino* nu1 = std::get<0>(nux[x]);
        neutrino* nu2 = std::get<1>(nux[x]); 
        this -> garbage.push_back(nu1); 
        this -> garbage.push_back(nu2); 

        for (size_t i(0); i < nu1 -> alternatives.size(); ++i){
            neutrino* nut1 = nu1 -> alternatives[i]; 
            neutrino* nut2 = nu2 -> alternatives[i]; 

            particle_template* p1 = new particle_template();
            p1 -> iadd(nut1);
            p1 -> iadd(nut1 -> lepton); 
            double w1 = p1 -> mass; 
            p1 -> iadd(nut1 -> bquark); 
            double t1 = p1 -> mass; 

            particle_template* p2 = new particle_template();
            p2 -> iadd(nut2);
            p2 -> iadd(nut2 -> lepton); 
            double w2 = p2 -> mass; 
            p2 -> iadd(nut2 -> bquark); 
            double t2 = p1 -> mass; 
            delete p1; delete p2; 

            double mx1 = std::pow((t1 - top_mass) / top_mass, 2) + std::pow((w1 - wboson_mass) / wboson_mass, 2);
            double mx2 = std::pow((t2 - top_mass) / top_mass, 2) + std::pow((w2 - wboson_mass) / wboson_mass, 2); 
            if (!n1){n1 = nut1; val1 = mx1;}
            if (!n2){n2 = nut2; val2 = mx2;}
            if (mx1 < val1){val1 = mx1; n1 = nut1;}
            if (mx2 < val2){val2 = mx2; n2 = nut2;}
        }
    }
    if (val1 > distance){n1 = nullptr;}
    if (val2 > distance){n2 = nullptr;}
    return {n1, n2};
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
    
    gr -> num_nodes    = this -> num_nodes; 
    gr -> event_index  = this -> index;
    gr -> event_weight = this -> weight; 
    gr -> preselection = this -> preselection; 
    gr -> graph_name   = new std::string(this -> name); 
    return gr; 
}

graph_template* graph_template::build(event_template* ev){
    event_t* data_ = &ev -> data; 
    data_ -> name = this -> name; 

    graph_template* gr = this -> clone(); 
    gr -> preselection = this -> preselection; 

    gr -> m_event = ev; 
    gr -> data = *data_; 
    gr -> filename  = ev -> filename; 
    gr -> meta_data = ev -> meta_data; 
    return gr; 
}

void graph_template::CompileEvent(){}
bool graph_template::PreSelection(){return true;}

