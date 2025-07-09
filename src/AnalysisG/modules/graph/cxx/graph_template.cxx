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

    auto pmc = [this](particle_template* pf) -> std::vector<double> {return {pf -> px, pf -> py, pf -> pz, pf -> e};}; 
    auto pmv = [this](std::vector<particle_template*> fx) -> std::vector<double> {
        std::vector<double> v = {0, 0, 0}; 
        for (size_t x(0); x < fx.size(); ++x){v[0] += fx[x] -> px; v[1] += fx[x] -> py; v[2] += fx[x] -> pz;}
        return v; 
    }; 

    auto masses = [this](std::vector<particle_template*> pbx) -> double {
        particle_template* p1 = new particle_template();
        for (size_t x(0); x < pbx.size(); ++x){p1 -> iadd(pbx[x]);}
        double mx = p1 -> mass; 
        delete p1; return mx*0.001;  
    };

    auto average = [this](std::vector<double> inx, double val) -> double {
        double f(inx.size() + !inx.size()); 
        double avg(0), std_(0); 
        for (size_t x(0); x < inx.size(); ++x){avg += inx[x];}
        avg = avg / f; 

        for (size_t x(0); x < inx.size(); ++x){std_ += std::pow(avg - inx[x], 2);}
        std_ = std::pow(std_ / f, 0.5); 
        return std::pow((val - avg)/std_, 2); 
    }; 
    auto error = [this](double t, double v, double e) -> bool{return std::abs(t - v)/t > e;}; 

    if (!particles.size()){return {nullptr, nullptr};}
    std::vector<double> pmc_ = pmv(particles); 
//    double mx = met*std::cos(phi) - pmc_[0]; 
//    double my = met*std::sin(phi) - pmc_[1]; 
    std::vector<double> metv = std::vector<double>({met}); //std::pow(mx*mx + my*my, 0.5)}); 
    std::vector<double> phiv = std::vector<double>({phi}); //std::atan2(my, mx)});
    std::vector<std::vector<particle_template*>> prt = {particles}; 

    #ifdef PYC_CUDA
    std::vector<std::pair<neutrino*, neutrino*>> nux; 
    nux = pyc::nusol::combinatorial(metv, phiv, prt, device, top_mass, wboson_mass, 0, perturb, steps); 
    #endif

    std::map<double, std::pair<neutrino*, neutrino*>> out; 
    for (size_t x(0); x < nux.size(); ++x){
        neutrino* nu1 = std::get<0>(nux[x]);
        neutrino* nu2 = std::get<1>(nux[x]); 
        this -> garbage.push_back(nu1); 
        this -> garbage.push_back(nu2); 
        std::vector<neutrino*> v1_ = nu1 -> alternatives; 
        std::vector<neutrino*> v2_ = nu2 -> alternatives; 
        v1_.push_back(nu1); v2_.push_back(nu2); 
        out[-2] = {nu1, nu2}; 
        //std::vector<int> vn; 
        //std::vector<double> w1_, w2_, t1_, t2_, avg; 
        //for (size_t i(0); i < v1_.size(); ++i){
        //    neutrino* n1 = v1_[i]; neutrino* n2 = v2_[i]; 
        //    particle_template* l1 = n1 -> lepton;
        //    particle_template* b1 = n1 -> bquark; 
        //    particle_template* l2 = n2 -> lepton;
        //    particle_template* b2 = n2 -> bquark; 
        //    double l1b1 = l1 -> DeltaR(b1); 
        //    double l1b2 = l1 -> DeltaR(b2); 
        //    double l2b1 = l2 -> DeltaR(b1);
        //    double l2b2 = l2 -> DeltaR(b2); 
        //    if (l1b1 > l1b2 || l2b2 > l2b1){continue;}
        //    double m1 = masses({l1, n1}); 
        //    if (error(wboson_mass*0.001, m1, distance)){continue;}
        //    double m2 = masses({l2, n2}); 
        //    if (error(wboson_mass*0.001, m2, distance)){continue;}
        //    double m3 = masses({l1, b1, n1}); 
        //    if (error(top_mass*0.001, m3, distance)){continue;}
        //    double m4 = masses({l2, b2, n2}); 
        //    if (error(top_mass*0.001, m4, distance)){continue;}
        //    w1_.push_back(m1); 
        //    w2_.push_back(m2); 
        //    t1_.push_back(m3); 
        //    t2_.push_back(m4); 
        //    avg.push_back(n1 -> min); 
        //    vn.push_back(i); 
        //}

        //for (size_t k(0); k < vn.size(); ++k){
        //    int i = vn[k]; 
        //    std::vector<double> pmc_b1 = pmc(v1_[i] -> bquark); 
        //    std::vector<double> pmc_b2 = pmc(v2_[i] -> bquark); 
        //    std::vector<double> pmc_l1 = pmc(v1_[i] -> lepton); 
        //    std::vector<double> pmc_l2 = pmc(v2_[i] -> lepton); 
        //    std::vector<double> pmc_n1 = pmc((particle_template*)v1_[i]); 
        //    std::vector<double> pmc_n2 = pmc((particle_template*)v2_[i]); 
        //    double dz = std::abs(pmc_[2] - (pmc_b1[2] + pmc_l1[2] + pmc_n1[2] + pmc_b2[2] + pmc_l2[2] + pmc_n2[2]))*0.001;
        //    double dy = std::abs(met*std::sin(phi) + pmc_[1] + (pmc_n1[1] + pmc_n2[1]))*0.001; 
        //    double dx = std::abs(met*std::cos(phi) + pmc_[0] + (pmc_n1[0] + pmc_n2[0]))*0.001; 
        //    double r  = std::pow(dx*dy*dz, 0.3); 
        //    r += average(w1_, w1_[k]) + average(w2_, w2_[k]);
        //    r += average(t1_, t1_[k]) + average(t2_, t2_[k]);
        //    r += average(avg, avg[k]); 
        //    out[r] = {v1_[i], v2_[i]}; 
        //}
    }
    if (!out.size()){out[-1] = {nullptr, nullptr};}
    return out.begin() -> second;
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

