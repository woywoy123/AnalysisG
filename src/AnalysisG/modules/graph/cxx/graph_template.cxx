#include <templates/graph_template.h>
#include <transform/cartesian-cuda.h>
#include <transform/polar-cuda.h>
#include <nusol/nusol-cuda.h>

graph_template::graph_template(){
    this -> op = new torch::TensorOptions(torch::kCPU);
    this -> name.set_getter(this -> set_name); 
    this -> name.set_object(this); 

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

bool graph_template::double_neutrino(
        double mass_top, double mass_wboson, 
        double top_perc, double w_perc, double distance, int steps 
){
    if (!this -> node_fx.count("D-pt")){return false;}
    if (!this -> node_fx.count("D-eta")){return false;}
    if (!this -> node_fx.count("D-phi")){return false;}
    if (!this -> node_fx.count("D-energy")){return false;}
    if (!this -> node_fx.count("D-is_lep")){return false;}
    if (!this -> node_fx.count("D-is_b")){return false;}
    if (!this -> graph_fx.count("D-met")){return false;}
    if (!this -> graph_fx.count("D-phi")){return false;}

    torch::Tensor is_b   = this -> node_fx["D-is_b"]; 
    torch::Tensor is_lep = this -> node_fx["D-is_lep"]; 
    torch::Tensor chk = (is_b.view({-1}).sum({-1}) >= 2) * (is_lep.view({-1}).sum({-1}) >= 2); 
    if (!chk.index({chk}).size({0})){return true;}

    torch::Tensor pt         = this -> node_fx["D-pt"].to(c10::kCUDA);
    torch::Tensor eta        = this -> node_fx["D-eta"].to(c10::kCUDA);
    torch::Tensor phi        = this -> node_fx["D-phi"].to(c10::kCUDA); 
    torch::Tensor energy     = this -> node_fx["D-energy"].to(c10::kCUDA);
    torch::Tensor pmc        = transform::cuda::PxPyPzE(pt, eta, phi, energy); 

    torch::Tensor pid        = torch::cat({is_lep, is_b}, {-1}).to(c10::kCUDA); 
    torch::Tensor edge_index = this -> m_topology.to(torch::kLong).to(c10::kCUDA); 
    torch::Tensor met        = this -> graph_fx["D-met"].to(c10::kCUDA); 
    torch::Tensor met_phi    = this -> graph_fx["D-phi"].to(c10::kCUDA);
    torch::Tensor batch      = torch::zeros_like(pt.view({-1})).to(torch::kLong); 
    torch::Tensor met_xy     = torch::cat({
            transform::cuda::Px(met, met_phi), 
            transform::cuda::Py(met, met_phi)
    }, {-1});

    // protection against overloading the cuda cores.
    //std::this_thread::sleep_for(std::chrono::microseconds(10)); 
    std::map<std::string, torch::Tensor> nus = nusol::cuda::combinatorial(
        edge_index, batch, pmc, pid, met_xy, mass_top, mass_wboson, 0.0, 
        top_perc, w_perc, distance, steps
    ); 
    
    torch::Tensor combi = nus["combi"].sum({-1}) > 0;
    if (!combi.index({combi}).size({0})){return true;}
    torch::Tensor lep1 = nus["combi"].index({combi, 2}).to(torch::kInt); 
    torch::Tensor lep2 = nus["combi"].index({combi, 3}).to(torch::kInt); 
    pmc.index_put_({lep1}, nus["nu_1f"] + pmc.index({lep1})); 
    pmc.index_put_({lep2}, nus["nu_2f"] + pmc.index({lep2}));
    pmc = transform::cuda::PtEtaPhiE(pmc).to(c10::kCPU);
    this -> node_fx["D-pt"]     = pmc.index({torch::indexing::Slice(), 0}).view({-1, 1}); 
    this -> node_fx["D-eta"]    = pmc.index({torch::indexing::Slice(), 1}).view({-1, 1}); 
    this -> node_fx["D-phi"]    = pmc.index({torch::indexing::Slice(), 2}).view({-1, 1}); 
    this -> node_fx["D-energy"] = pmc.index({torch::indexing::Slice(), 3}).view({-1, 1}); 
    return true;
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
    gr -> graph_name   = new std::string(this -> name); 
    return gr; 
}

graph_template* graph_template::build(event_template* ev){
    event_t* data_ = &ev -> data; 
    data_ -> name = this -> name; 

    graph_template* gr = this -> clone(); 
    gr -> m_event = ev; 
    gr -> data = *data_; 
    gr -> filename = ev -> filename; 
    return gr; 
}

void graph_template::CompileEvent(){}
