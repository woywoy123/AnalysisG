#include <metrics/accuracy.h>
#include <metrics/pagerank.h>

std::vector<particle_template*> accuracy_metric::build_top(std::map<int, std::map<int, particle_template*>>* mx){
    std::vector<particle_template*> out = {}; 
    std::map<double, particle_template*> tops_ = {}; 
    std::map<int, std::map<int, particle_template*>>::iterator itr = mx -> begin(); 
    for (; itr != mx -> end(); ++itr){
        std::vector<particle_template*> tmp; 
        std::map<int, particle_template*>::iterator itx = itr -> second.begin(); 
        for (; itx != itr -> second.end(); ++itx){tmp.push_back(itx -> second);}
        particle_template* px = nullptr; 
        this -> sum(&tmp, &px); 
        double mass = px -> mass; 
        if (tops_.count(mass)){continue;}
        tops_[mass] = px;
        out.push_back(px); 
    }
    return out; 
}

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
    
    std::vector<particle_template*> ptx = make_particle(&pt, &eta, &phi, &e);
    std::vector<std::vector<int>> is_lep = mtx -> get<std::vector<std::vector<int>>>(graph_enum::data_node, "is_lep");
    std::vector<std::vector<int>> is_bq  = mtx -> get<std::vector<std::vector<int>>>(graph_enum::data_node, "is_b");
    for (size_t x(0); x < ptx.size(); ++x){
        if (is_lep[x][0] > 0){ptx[x] -> pdgid = 11;}
        if ( is_bq[x][0] > 0){ptx[x] -> pdgid = 5;}
    }
    return ptx;
}


float edge_f1(std::vector<int>* pred, std::vector<int>* truth){
    if (!pred->size()) { return 0.0; }
    float tp = 0, tn = 0;
    float fp = 0, fn = 0;
    for (size_t i(0); i < pred -> size(); ++i) {
        int t = truth -> at(i); 
        int p = pred -> at(i); 
        tp += (t == 1) * (p == 1); 
        tn += (t == 1) * (p == 0); 
        fp += (t == 0) * (p == 1); 
        fn += (t == 0) * (p == 0); 
    }
    return (tp + fn) / (tp + tn + fn + fp);
}

int get_maxidx(std::vector<float>* acx){
    int idx = 0; 
    float v = acx -> at(0); 
    for (size_t x(0); x < acx -> size(); ++x){
        if (acx -> at(x) < v){continue;}
        v = acx -> at(x); idx = x; 
    }
    return idx; 
}


