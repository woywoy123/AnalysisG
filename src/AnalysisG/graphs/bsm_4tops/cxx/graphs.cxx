#include <bsm_4tops/graph_features.h>
#include <bsm_4tops/node_features.h>
#include <bsm_4tops/edge_features.h>
#include <bsm_4tops/graphs.h>
#include <bsm_4tops/event.h>

// ---------------- GRAPH-TOPS ------------------- //
graph_tops::graph_tops(){this -> name = "graph_tops";}
graph_tops::~graph_tops(){}
graph_template* graph_tops::clone(){return new graph_tops();}

void graph_tops::CompileEvent(){
    bsm_4tops* event = this -> get_event<bsm_4tops>(); 
    this -> define_particle_nodes(&event -> Tops); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, bsm_4tops>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int, bsm_4tops>(event , num_tops    , "ntops"); 

    this -> add_node_truth_feature<int, particle_template>(top_node, "top_node"); 
    this -> add_node_truth_feature<int, particle_template>(res_node, "res_node");
    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_et , "met"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_phi, "phi"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long, bsm_4tops>(event, event_number, "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt    , "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta   , "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi   , "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 
    this -> add_node_data_feature<double, particle_template>(charge, "charge"); 
}

// ---------------- GRAPH-TOPS ------------------- //
graph_children::graph_children(){this -> name = "graph_children";}
graph_children::~graph_children(){}
graph_template* graph_children::clone(){return new graph_children();}

void graph_children::CompileEvent(){
    bsm_4tops* event = this -> get_event<bsm_4tops>(); 
    this -> define_particle_nodes(&event -> Children); 
    this -> define_topology(fulltopo); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, bsm_4tops>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int, bsm_4tops>(event, num_neutrino, "n_nu"); 
    this -> add_graph_truth_feature<int, bsm_4tops>(event, num_lepton, "n_lep"); 
    this -> add_graph_truth_feature<int, bsm_4tops>(event, num_tops , "ntops"); 

    this -> add_node_truth_feature<int, particle_template>(top_node, "top_node"); 
    this -> add_node_truth_feature<int, particle_template>(res_node, "res_node"); 

    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 
    this -> add_edge_truth_feature<int, particle_template>(top_edge, "top_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_et, "met"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_phi, "phi"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_quark, "num_jets"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_children_leps, "num_leps"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long, bsm_4tops>(event, event_number, "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt    , "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta   , "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi   , "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 
    this -> add_node_data_feature<double, particle_template>(charge, "charge"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
    this -> add_node_data_feature<int, particle_template>(is_neutrino, "is_nu");
}

// ---------------- GRAPH-TRUTHJETS ------------------- //
graph_truthjets::graph_truthjets(){this -> name = "graph_truthjets";}
graph_truthjets::~graph_truthjets(){}
graph_template* graph_truthjets::clone(){return new graph_truthjets();}

void graph_truthjets::CompileEvent(){
    bsm_4tops* event = this -> get_event<bsm_4tops>(); 
    
    std::vector<particle_template*> nodes = {}; 
    std::vector<particle_template*> ch = event -> Children;
    for (size_t x(0); x < ch.size(); ++x){
        if (!ch[x] -> is_lep && !ch[x] -> is_nu){continue;}
        nodes.push_back(ch[x]); 
    }
    nodes.insert(nodes.end(), event -> TruthJets.begin(), event -> TruthJets.end()); 
    this -> define_particle_nodes(&nodes); 
    this -> define_topology(fulltopo); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, bsm_4tops>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_neutrino, "n_nu"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_lepton  , "n_lep"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_tops    , "ntops"); 

    this -> add_node_truth_feature<int, particle_template>(top_node, "top_node"); 
    this -> add_node_truth_feature<int, particle_template>(res_node, "res_node"); 

    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 
    this -> add_edge_truth_feature<int, particle_template>(top_edge, "top_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_et, "met"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_phi, "phi"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_truthjets, "num_jets"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_children_leps, "num_leps"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long, bsm_4tops>(event, event_number, "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt, "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta, "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi, "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 
    this -> add_node_data_feature<double, particle_template>(charge, "charge"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
    this -> add_node_data_feature<int, particle_template>(is_neutrino, "is_nu");
}

// ---------------- GRAPH-TRUTHJETS (No Neutrino) ------------------- //
graph_truthjets_nonu::graph_truthjets_nonu(){this -> name = "graph_truthjets_nonu";}
graph_truthjets_nonu::~graph_truthjets_nonu(){}
graph_template* graph_truthjets_nonu::clone(){return new graph_truthjets_nonu();}

void graph_truthjets_nonu::CompileEvent(){
    bsm_4tops* event = this -> get_event<bsm_4tops>(); 
    
    std::vector<particle_template*> nodes = {}; 
    std::vector<particle_template*> ch = event -> Children;
    for (size_t x(0); x < ch.size(); ++x){
        if (!ch[x] -> is_lep || ch[x] -> is_nu){continue;}
        nodes.push_back(ch[x]); 
    }
    nodes.insert(nodes.end(), event -> TruthJets.begin(), event -> TruthJets.end()); 
    this -> define_particle_nodes(&nodes); 
    this -> define_topology(fulltopo); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, bsm_4tops>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_lepton  , "n_lep"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_tops    , "ntops"); 

    this -> add_node_truth_feature<int, particle_template>(top_node, "top_node"); 
    this -> add_node_truth_feature<int, particle_template>(res_node, "res_node"); 

    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 
    this -> add_edge_truth_feature<int, particle_template>(top_edge, "top_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_et, "met"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_phi, "phi"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_truthjets, "num_jets"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_children_leps, "num_leps"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long, bsm_4tops>(event, event_number, "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt, "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta, "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi, "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 
    this -> add_node_data_feature<double, particle_template>(charge, "charge"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
}


// ---------------- GRAPH-JETS ------------------- //
graph_jets::graph_jets(){this -> name = "graph_jets";}
graph_jets::~graph_jets(){}
graph_template* graph_jets::clone(){return new graph_jets();}

bool graph_jets::PreSelection(){
    bsm_4tops* evn = this -> get_event<bsm_4tops>();
    std::vector<particle_template*> leptons; 
    for (size_t x(0); x < evn -> Electrons.size(); ++x){leptons.push_back(evn -> Electrons[x]);} 
    for (size_t x(0); x < evn -> Muons.size(); ++x){leptons.push_back(evn -> Muons[x]);} 
    if (leptons.size() != 2){return false;}
    return true; //leptons[0] -> charge == leptons[1] -> charge; 
}

void graph_jets::CompileEvent(){
    bsm_4tops* event = this -> get_event<bsm_4tops>(); 
    
    std::vector<particle_template*> nodes = {}; 
    std::vector<particle_template*> ch = event -> Children;
    for (size_t x(0); x < ch.size(); ++x){
        if (!ch[x] -> is_lep && !ch[x] -> is_nu){continue;}
        nodes.push_back(ch[x]); 
    }
    nodes.insert(nodes.end(), event -> Jets.begin(), event -> Jets.end()); 
    this -> define_particle_nodes(&nodes); 
    this -> define_topology(fulltopo); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, bsm_4tops>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_lepton  , "n_lep"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_tops    , "ntops"); 

    this -> add_node_truth_feature<int, particle_template>(top_node, "top_node"); 
    this -> add_node_truth_feature<int, particle_template>(res_node, "res_node"); 

    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 
    this -> add_edge_truth_feature<int, particle_template>(top_edge, "top_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_et       , "met"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_phi      , "phi"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_jets         , "num_jets"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_children_leps, "num_leps"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, event_weight     , "weight"); 
    this -> add_graph_data_feature<long  , bsm_4tops>(event, event_number     , "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt    , "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta   , "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi   , "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 
    this -> add_node_data_feature<double, particle_template>(charge, "charge"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
}

// ---------------- GRAPH-JETS (No Neutrino) ------------------- //
graph_jets_nonu::graph_jets_nonu(){this -> name = "graph_jets_nonu";}
graph_jets_nonu::~graph_jets_nonu(){}
graph_template* graph_jets_nonu::clone(){return new graph_jets_nonu();}

void graph_jets_nonu::CompileEvent(){
    bsm_4tops* event = this -> get_event<bsm_4tops>(); 
    
    std::vector<particle_template*> nodes = {}; 
    std::vector<particle_template*> ch = event -> Children;
    for (size_t x(0); x < ch.size(); ++x){
        if (!ch[x] -> is_lep || ch[x] -> is_nu){continue;}
        nodes.push_back(ch[x]); 
    }
    nodes.insert(nodes.end(), event -> Jets.begin(), event -> Jets.end()); 
    this -> define_particle_nodes(&nodes); 
    this -> define_topology(fulltopo); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, bsm_4tops>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_lepton  , "n_lep"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_tops    , "ntops"); 

    this -> add_node_truth_feature<int, particle_template>(top_node, "top_node"); 
    this -> add_node_truth_feature<int, particle_template>(res_node, "res_node"); 

    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 
    this -> add_edge_truth_feature<int, particle_template>(top_edge, "top_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_et, "met"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_phi, "phi"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_jets, "num_jets"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_children_leps, "num_leps"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long, bsm_4tops>(event, event_number, "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt, "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta, "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi, "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 
    this -> add_node_data_feature<double, particle_template>(charge, "charge"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
}


// ---------------- GRAPH-JETS-Detector leptons (with Neutrino) ------------------- //
graph_jets_detector_lep::graph_jets_detector_lep(){this -> name = "graph_jets_detector_lep";}
graph_jets_detector_lep::~graph_jets_detector_lep(){}
graph_template* graph_jets_detector_lep::clone(){return new graph_jets_detector_lep();}

void graph_jets_detector_lep::CompileEvent(){
    bsm_4tops* event = this -> get_event<bsm_4tops>(); 
    
    std::vector<particle_template*> nodes = {}; 
    nodes.insert(nodes.end(), event -> Muons.begin(), event -> Muons.end()); 
    nodes.insert(nodes.end(), event -> Electrons.begin(), event -> Electrons.end()); 
    std::vector<particle_template*> ch = event -> Children;
    for (size_t x(0); x < ch.size(); ++x){
        if (!ch[x] -> is_nu){continue;}
        nodes.push_back(ch[x]); 
    }
    nodes.insert(nodes.end(), event -> Jets.begin(), event -> Jets.end()); 
    this -> define_particle_nodes(&nodes); 
    this -> define_topology(fulltopo); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, bsm_4tops>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_lepton  , "n_lep"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_tops    , "ntops"); 

    this -> add_node_truth_feature<int, particle_template>(top_node, "top_node"); 
    this -> add_node_truth_feature<int, particle_template>(res_node, "res_node"); 

    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 
    this -> add_edge_truth_feature<int, particle_template>(top_edge, "top_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_et  , "met"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_phi , "phi"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_jets    , "num_jets"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_leps    , "num_leps"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long, bsm_4tops>(event, event_number  , "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt    , "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta   , "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi   , "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 
    this -> add_node_data_feature<double, particle_template>(charge, "charge"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
}

// ---------------- GRAPH-JETS-Detector leptons (without Neutrino) ------------------- //
graph_detector::graph_detector(){this -> name = "graph_detector";}
graph_detector::~graph_detector(){}
graph_template* graph_detector::clone(){
    graph_detector* gx = new graph_detector();
    gx -> force_match  = this -> force_match;
    gx -> num_cuda     = this -> num_cuda;
    gx -> preselection = this -> preselection; 
    return gx;
}

bool graph_detector::PreSelection(){
    std::vector<particle_template*> leptons; 
    bsm_4tops* evn = this -> get_event<bsm_4tops>();
    for (size_t x(0); x < evn -> Electrons.size(); ++x){leptons.push_back(evn -> Electrons[x]);} 
    for (size_t x(0); x < evn -> Muons.size(); ++x){leptons.push_back(evn -> Muons[x]);} 
    return leptons.size() == 2; 
}

void graph_detector::CompileEvent(){
    auto assign = [this](particle_template* prt, neutrino* nu, std::vector<int>* idx, bool* res) -> bool{
        std::string hash_ = prt -> hash; 
        bool is_res = false;
        std::vector<int> top_idx = {}; 
        std::string type_ = prt -> type; 
        if (type_ == "jet"){top_idx = ((jet*)prt) -> top_index;              is_res = ((jet*)prt) -> from_res;     }
        if (type_ == "el"){top_idx.push_back(((electron*)prt) -> top_index); is_res = ((electron*)prt) -> from_res;}
        if (type_ == "mu"){top_idx.push_back(((muon*)prt) -> top_index);     is_res = ((muon*)prt) -> from_res;    }
        idx -> insert(idx -> end(), top_idx.begin(), top_idx.end()); 
        *res = is_res;

        if (!nu){return false;}
        bool found = false;
        found += (hash_ == std::string(nu -> bquark -> hash)); 
        found += (hash_ == std::string(nu -> lepton -> hash)); 
        if (!found){return false;}
        if (is_res){nu -> from_res = is_res;}
        for (size_t x(0); x < top_idx.size(); ++x){nu -> top_index.push_back(top_idx[x]);}
        top_idx.clear(); 
        this -> unique_key(&nu -> top_index, &top_idx); 
        nu -> top_index = top_idx; 
        return found; 
    }; 

    auto deassign = [this](std::map<std::string, particle_template*> hprt, int topx){
        std::map<std::string, particle_template*>::iterator itr = hprt.begin();
        for (; itr != hprt.end(); ++itr){
            particle_template* prt = itr -> second;
            std::string type_ = prt -> type; 
            std::vector<int>* v = nullptr; 
            if (type_ == "jet" ){v =      &((jet*)prt) -> top_index;}
            if (type_ == "el"  ){     ((electron*)prt) -> top_index = -1;}
            if (type_ == "mu"  ){         ((muon*)prt) -> top_index = -1;}
            if (type_ == "nunu"){v = &((neutrino*)prt) -> top_index;}
            if (!v){continue;}
            std::vector<int> w = {}; 
            for (size_t x(0); x < v -> size(); ++x){
                if (v -> at(x) == topx){continue;}
                w.push_back(v -> at(x)); 
            }
            v -> clear(); *v = w; 
        }
    }; 

    bsm_4tops* event = this -> get_event<bsm_4tops>();
    std::vector<particle_template*> nodes = {}; 
    nodes.insert(nodes.end(), event -> Muons.begin()    , event -> Muons.end()); 
    nodes.insert(nodes.end(), event -> Electrons.begin(), event -> Electrons.end()); 
    nodes.insert(nodes.end(), event -> Jets.begin()     , event -> Jets.end()); 


    std::map<int, particle_template*> nus = {}; 
    std::vector<particle_template*> ch = event -> Children;
    for (size_t x(0); x < ch.size(); ++x){
        if (!ch[x] -> is_nu){continue;}
        top_children* chx = (top_children*)ch[x]; 
        nus[chx -> top_index] = ch[x]; 
    }

    std::cout << "-----" << std::endl;
    neutrino* nu1 = nullptr; neutrino* nu2 = nullptr;
    if ((event -> Muons.size() + event -> Electrons.size()) == 2){
        std::pair<particle_template*, particle_template*> nux;
        std::string cu = "cuda:" + std::to_string(this -> threadIdx % this -> num_cuda);  
        nux = this -> double_neutrino(nodes, event -> met, event -> phi, cu, 172.62*1000.0, 80.385*1000.0, 5e-1, 1e-1, 100);
        nu1 = (neutrino*)std::get<0>(nux); nu2 = (neutrino*)std::get<1>(nux); 
    }

    std::map<std::string, particle_template*> hash_map_res; 
    std::map<int, std::map<std::string, particle_template*>> hash_map_top; 
    for (size_t x(0); x < nodes.size(); ++x){
        std::vector<int> top_i = {}; 
        bool is_res = false; 
        particle_template* prt = nodes[x]; 
        bool f = assign(prt, nu1, &top_i, &is_res); 
        for (size_t k(0); k < top_i.size(); ++k){
            int ti = top_i[k]; 
            hash_map_top[ti][prt -> hash] = prt;
            if (is_res){hash_map_res[prt -> hash] = prt;}
            if (!f){continue;}
            //hash_map_top[ti][nu1 -> hash] = nu1;
            if (is_res){hash_map_res[nu1 -> hash] = nu1;}
        }

        top_i = {}; 
        is_res = false; 
        f = assign(prt, nu2, &top_i, &is_res);
        for (size_t k(0); k < top_i.size(); ++k){
            int ti = top_i[k]; 
            hash_map_top[ti][prt -> hash] = prt;
            if (is_res){hash_map_res[prt -> hash] = prt;}
            if (!f){continue;}
            //hash_map_top[ti][nu2 -> hash] = nu2;
            if (is_res){hash_map_res[nu2 -> hash] = nu2;}
        }
    }


    auto lamb = [this](std::map<std::string, particle_template*> pbx, particle_template* nx) -> double {
        particle_template* p1 = new particle_template();
        std::map<std::string, particle_template*>::iterator itx = pbx.begin(); 
        for (; itx != pbx.end(); ++itx){p1 -> iadd(itx -> second);}
        p1 -> iadd(nx); 
        double mx = p1 -> mass; 
        delete p1; return mx;  
    };


    std::string hb, hl, hn; 
    std::map<int, std::map<std::string, particle_template*>>::iterator itx; 
    for (itx = hash_map_top.begin(); itx != hash_map_top.end(); ++itx){
        if (nus.count(itx -> first)){
            std::cout << lamb(itx -> second, nus[itx -> first]) << std::endl;
            continue; 
        }

        if (!nu1 || !nu2 || itx -> first < 0){deassign(itx -> second, itx -> first); continue;}
        hb = nu1 -> bquark -> hash; hl = nu1 -> lepton -> hash; hn = nu1 -> hash; 
        if (itx -> second.count(hb) && itx -> second.count(hl) && itx -> second.count(hn)){continue;}
        hb = nu2 -> bquark -> hash; hl = nu2 -> lepton -> hash; hn = nu2 -> hash; 
        if (itx -> second.count(hb) && itx -> second.count(hl) && itx -> second.count(hn)){continue;}
        deassign(itx -> second, itx -> first);
    }
    if (nu1){nodes.push_back(nu1);}
    if (nu2){nodes.push_back(nu2);}

    this -> define_particle_nodes(&nodes); 
    this -> define_topology(fulltopo); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, bsm_4tops>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_lepton  , "n_lep"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_tops    , "ntops"); 

    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 
    this -> add_edge_truth_feature<int, particle_template>(top_edge, "top_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_et  , "met"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_phi , "phi"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_jets    , "num_jets"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_leps    , "num_leps"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long  , bsm_4tops>(event, event_number, "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt , "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta, "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi, "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 
    this -> add_node_data_feature<double, particle_template>(charge, "charge"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
}
