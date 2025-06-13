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

    auto lamb = [this](std::vector<particle_template*> pbx) -> double {
        particle_template* p1 = new particle_template();
        for (size_t x(0); x < pbx.size(); ++x){p1 -> iadd(pbx[x]);}
        double mx = p1 -> mass; 
        delete p1; return mx;  
    };

    auto pmc = [this](particle_template* pf, std::string dim) -> double{
        if (dim == "x"){return pf -> px*0.001;}
        if (dim == "y"){return pf -> py*0.001;}
        if (dim == "z"){return pf -> pz*0.001;}
        return 0; 
    }; 

    auto cross = [this](double x1, double y1, double z1, double x2, double y2, double z2) -> std::vector<double> {
        double xi = x1 * z2 - z1 * y2; 
        double yi = z1 * x2 - x1 * z2; 
        double zi = x1 * y2 - y1 * x2; 
        double di = std::pow(xi*xi + yi*yi + zi*zi, 0.5); 
        xi = xi / di; yi = yi / di; zi = zi / di; 
        return {xi, yi, zi, di}; 
    }; 

    auto angle = [this, pmc, cross](
            neutrino* nu1, neutrino* nu2, double met_, double phi_, std::vector<particle_template*>* nodes_
    ) -> std::vector<double> {
        // define the plane of neutrino 1 and 2 - W boson plane
        std::vector<double> nx1 = cross(pmc(nu1, "x"), pmc(nu1, "y"), pmc(nu1, "z"), pmc(nu1 -> lepton, "x"), pmc(nu1 -> lepton, "y"), pmc(nu1 -> lepton, "z")); 
        std::vector<double> nx2 = cross(pmc(nu2, "x"), pmc(nu2, "y"), pmc(nu2, "z"), pmc(nu2 -> lepton, "x"), pmc(nu2 -> lepton, "y"), pmc(nu2 -> lepton, "z")); 

        // define the plane of the bb projectile 
        std::vector<double> bxn = cross(pmc(nu1 -> bquark, "x"), pmc(nu1 -> bquark, "y"), pmc(nu1 -> bquark, "z"), pmc(nu2 -> bquark, "x"), pmc(nu2 -> bquark, "y"), pmc(nu2 -> bquark, "z")); 

        // define the met circle plane
        double mx_ = met_ * std::cos(phi_)*0.001;
        double my_ = met_ * std::sin(phi_)*0.001;
        double mz_ = 0; 

        // define the particle plane
        double x_(0), y_(0), z_(0);
        for (size_t x(0); x < nodes_ -> size(); ++x){z_ += pmc(nodes_ -> at(x),"z");}
        x_ += met_ * std::cos(phi_)*0.001; 
        y_ += met_ * std::sin(phi_)*0.001;  
        z_ -= pmc(nu1 -> bquark, "z") + pmc(nu1 -> lepton, "z") + pmc(nu1, "z"); 
        z_ -= pmc(nu2 -> bquark, "z") + pmc(nu2 -> lepton, "z") + pmc(nu2, "z"); 

        // plane of nunu
        std::vector<double> n1xn2 = cross(nx1[0], nx1[1], nx1[2], nx2[0], nx2[1], nx2[2]); 

//        // plane of met X particles cross bb system 
//        std::vector<double> nu_bb = cross(bxn[0], bxn[1], bxn[2], n1xn2[0], n1xn2[1], n1xn2[2]); 

        std::vector<double> vxp = cross(x_, y_, z_, n1xn2[0], n1xn2[1], n1xn2[2]); 

        double dg = (180.0/3.141592653589793238463); 
        return {std::acos(vxp[2])*dg, std::atan2(vxp[1], vxp[0])*dg, vxp[2], std::abs(vxp[3])}; 
    }; 

    if (!particles.size()){return {nullptr, nullptr};}
    std::vector<std::pair<neutrino*, neutrino*>> nux; 
    std::vector<double> metv = std::vector<double>({met}); 
    std::vector<double> phiv = std::vector<double>({phi});
    std::vector<std::vector<particle_template*>> prt = {particles}; 

    //distance = 0; 
    //for (size_t x(0); x < particles.size(); ++x){
    //    distance += pmc(particles.at(x), "z");  
    //}

    #ifdef PYC_CUDA
    nux = pyc::nusol::combinatorial(metv, phiv, prt, device, top_mass, wboson_mass, distance, perturb, steps); 
    #endif

    neutrino* n1 = nullptr; 
    neutrino* n2 = nullptr; 
    for (size_t x(0); x < nux.size(); ++x){
        double mx(0); 
        neutrino* nu1 = std::get<0>(nux[x]);
        neutrino* nu2 = std::get<1>(nux[x]); 
        this -> garbage.push_back(nu1); 
        this -> garbage.push_back(nu2); 

        std::vector<neutrino*> v1 = nu1 -> alternatives; 
        std::vector<neutrino*> v2 = nu2 -> alternatives; 

        v1.push_back(nu1); 
        v2.push_back(nu2); 

        double xnt = 0; 
        double agl = 0; 
        double mxi = 0; 
        for (size_t i(0); i < v1.size(); ++i){
            std::vector<double> vx = angle(v1[i], v2[i], met, phi, &particles);
            agl += vx[0]; xnt += 1.0; mxi += v1[i] -> min; 
            double t1 = lamb({v1[i], v1[i] -> lepton, v1[i] -> bquark}); 
            double t2 = lamb({v2[i], v2[i] -> lepton, v2[i] -> bquark}); 
            double w1 = lamb({v1[i], v1[i] -> lepton}); 
            double w2 = lamb({v2[i], v2[i] -> lepton}); 
            std::cout << "-> " << t1 << " " << t2 << " " << vx[0] << " " << vx[1] << " " << vx[2] << " " << vx[3] << " " << v1[i] -> min << std::endl;
            n1 = v1[i]; n2 = v2[i]; 
        }
        
        agl = agl / (xnt + (!xnt)); 
        mxi = mxi / (xnt + (!xnt)); 
        for (size_t i(0); i < v1.size(); ++i){
            //if (v1[i] -> min > mxi){continue;}
            std::vector<double> vx = angle(v1[i], v2[i], met, phi, &particles);
            double mv = abs(vx[0] - agl); 
            if (mx > mv){continue;}
            double t1 = lamb({v1[i], v1[i] -> lepton, v1[i] -> bquark}); 
            double t2 = lamb({v2[i], v2[i] -> lepton, v2[i] -> bquark}); 
            double w1 = lamb({v1[i], v1[i] -> lepton}); 
            double w2 = lamb({v2[i], v2[i] -> lepton}); 
            if (std::pow((t1 - top_mass)/top_mass, 2) > distance){continue;} 
            if (std::pow((t2 - top_mass)/top_mass, 2) > distance){continue;} 
            if (std::pow((w1 - wboson_mass)/wboson_mass, 2) > distance){continue;} 
            if (std::pow((w2 - wboson_mass)/wboson_mass, 2) > distance){continue;} 
            mx = mv; n1 = v1[i]; n2 = v2[i]; 
        }

    }
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

