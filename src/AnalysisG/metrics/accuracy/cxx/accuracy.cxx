#include <metrics/accuracy.h>

accuracy_metric::~accuracy_metric(){}
accuracy_metric* accuracy_metric::clone(){return new accuracy_metric();}
accuracy_metric::accuracy_metric(){this -> name = "accuracy";}

std::vector<long> accuracy_metric::event_index(metric_t* mtx){
    return mtx -> get<std::vector<long>>( graph_enum::batch_events, "index");
}

std::vector<long> accuracy_metric::batch_index(metric_t* mtx){
    return mtx -> get<std::vector<long>>( graph_enum::batch_index, "index");
}

std::vector<std::vector<int>> accuracy_metric::edge_index(metric_t* mtx){
    return mtx -> get<std::vector<std::vector<int>>>( graph_enum::edge_index, "index");
}

std::vector<std::vector<int>> accuracy_metric::top_edge_truth(metric_t* mtx){
    return mtx -> get<std::vector<std::vector<int>>>( graph_enum::truth_edge, "top_edge");
}

std::vector<std::vector<float>> accuracy_metric::top_edge_score(metric_t* mtx){
    return mtx -> get<std::vector<std::vector<float>>>( graph_enum::pred_extra, "top_edge_score");
}

std::vector<std::vector<float>> accuracy_metric::ntops_score(metric_t* mtx){
    return mtx -> get<std::vector<std::vector<float>>>( graph_enum::pred_extra, "ntops_score");
}

std::vector<std::vector<int>> accuracy_metric::ntops_truth(metric_t* mtx){
    return mtx -> get<std::vector<std::vector<int>>>( graph_enum::truth_graph, "ntops");
}

std::vector<particle_template*> accuracy_metric::build_particles(metric_t* mtx){
    std::vector<std::vector<double>> e   = mtx -> get<std::vector<std::vector<double>>>(graph_enum::data_node, "energy");
    std::vector<std::vector<double>> pt  = mtx -> get<std::vector<std::vector<double>>>(graph_enum::data_node, "pt");
    std::vector<std::vector<double>> phi = mtx -> get<std::vector<std::vector<double>>>(graph_enum::data_node, "phi");
    std::vector<std::vector<double>> eta = mtx -> get<std::vector<std::vector<double>>>(graph_enum::data_node, "eta");
    return make_particle(&pt, &eta, &phi, &e);
}

void accuracy_metric::define_variables(metric_t* mtx){
    this -> output_path = "./ProjectName/metrics/accuracy/epoch-" + std::to_string(mtx -> epoch) + "/";  
    this -> create_path(this -> output_path); 
    this -> output_path += "kfold-" + std::to_string(mtx -> kfold) + ".root"; 

    this -> register_output(&this -> ntops_prd); 
    this -> register_output(&this -> ntops_tru); 
    this -> register_output(&this -> ntops_scores); 
}

void accuracy_metric::end(){}

void accuracy_metric::define_metric(metric_t* mtx){
    auto maxv =[](std::vector<float>* acx) -> int {
        int idx = 0; 
        float v = acx -> at(0); 
        for (size_t x(0); x < acx -> size(); ++x){
            if (acx -> at(x) < v){continue;}
            v = acx -> at(x); idx = x; 
        }
        return idx; 
    }; 

    std::vector<std::vector<int>>   edge_ix = this -> edge_index(mtx); 
    std::vector<std::vector<float>> edge_sc = this -> top_edge_score(mtx); 
    std::vector<std::vector<int>>   edge_tr = this -> top_edge_truth(mtx); 
    std::vector<std::vector<int>>   ntops_tr = this -> ntops_truth(mtx); 
    std::vector<std::vector<float>> ntops_sc = this -> ntops_score(mtx);
    std::vector<particle_template*> ptx = this -> build_particles(mtx); 

    std::vector<long> btch_ix = this -> batch_index(mtx); 
    std::vector<long> evnt_ix = this -> event_index(mtx); 
        
    int num_nodes = 0; 
    for (size_t x(0); x < evnt_ix.size(); ++x){
        event_idx evnt; 
        evnt.edge_index.push_back({}); 
        evnt.edge_index.push_back({}); 
        for (size_t y(0); y < edge_ix[0].size(); ++y){
            if (btch_ix[edge_ix[0][y]] != evnt_ix[x]){continue;}
            evnt.edge_index[0].push_back(edge_ix[0][y] - num_nodes); 
            evnt.edge_index[1].push_back(edge_ix[1][y] - num_nodes); 
            evnt.top_edge_score.push_back(edge_sc[y]); 
            evnt.top_edge_truth.push_back(edge_tr[y][0]); 
            evnt.top_edge_pred.push_back(maxv(&edge_sc[y])); 
        }
        for (size_t y(0); y < btch_ix.size(); ++y){
            if (evnt_ix[x] != btch_ix[y]){continue;}
            evnt.ptx.push_back(ptx[y]); 
            num_nodes += 1;
        }
        evnt.n_tops_truth = ntops_tr[x][0]; 
        evnt.n_tops_score = ntops_sc[x]; 
        evnt.n_tops_pred  = maxv(&evnt.n_tops_score); 
        evnt.file = mtx -> get_filename(x); 
        evnt.process_ix = int(process_sample(evnt.file)); 
        this -> pagerank(&evnt); 
    }
}



