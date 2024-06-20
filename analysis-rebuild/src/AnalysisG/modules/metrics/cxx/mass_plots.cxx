#include <transform/cartesian-cuda.h>
#include <physics/physics-cuda.h>
#include <graph/graph-cuda.h>
#include <metrics/metrics.h>

void metrics::dump_mass_plots(){}

void metrics::build_th1f_mass(std::string var_name, graph_enum typ, int kfold){
   
    analytics_t* an = &this -> registry[kfold]; 

    std::string title = ""; 
    std::map<mode_enum, std::map<std::string, TH1F*>>* type_ = nullptr; 
    switch(typ){
        case graph_enum::truth_edge: type_ = &an -> truth_mass_edge; title = "truth"; break; 
        case graph_enum::data_edge:  type_ = &an -> pred_mass_edge;  title = "predicted"; break; 
        default: return; 
    }

    std::string tr = var_name + " - " + title + " (training) Mass " + std::to_string(kfold+1) + "-fold"; 
    (*type_)[mode_enum::training][var_name] = new TH1F(tr.c_str(), "training", 100, 0, 400); 

    std::string va = var_name + " - " + title + " (validation) Mass" + std::to_string(kfold+1) + "-fold"; 
    (*type_)[mode_enum::validation][var_name] = new TH1F(va.c_str(), "validation", 100, 0, 400); 

    std::string ev = var_name + " - " + title + " (evaluation) Mass" + std::to_string(kfold+1) + "-fold"; 
    (*type_)[mode_enum::evaluation][var_name] = new TH1F(ev.c_str(), "evaluation", 100, 0, 400); 
}

void metrics::add_th1f_mass(
        std::map<std::string, torch::Tensor*>* node_feats, 
        torch::Tensor* edge_index, torch::Tensor* truth, torch::Tensor* pred, 
        int kfold, mode_enum mode
){

    torch::Tensor pmc = torch::cat({
            *(*node_feats)["pt"] , *(*node_feats)["eta"], 
            *(*node_feats)["phi"], *(*node_feats)["energy"]
    }, {-1}); 
    pmc = transform::cuda::PxPyPzE(pmc);

    std::map<std::string, std::vector<torch::Tensor>> pred_mass = graph::cuda::edge_aggregation(edge_index -> to(torch::kLong), *pred, pmc); 
    torch::Tensor pred_mass_cu = physics::cuda::M(pred_mass["1"][1]); 
    pred_mass_cu = pred_mass_cu.index({(pred_mass_cu > 0).view({-1})}); 
    pred_mass_cu = (pred_mass_cu/1000).view({-1}).to(torch::kCPU); 
    std::vector<float> v(pred_mass_cu.data_ptr<float>(), pred_mass_cu.data_ptr<float>() + pred_mass_cu.numel());

    analytics_t* an = &this -> registry[kfold]; 
    std::map<mode_enum, std::map<std::string, TH1F*>>* type_ = &an -> pred_mass_edge; 
    for (size_t x(0); x < v.size(); ++x){(*type_)[mode]["top_edge"] -> Fill(v[x]);} 
} 


