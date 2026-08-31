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
    
    std::vector<particle_template*> ptx = make_particle(&pt, &eta, &phi, &e);
    std::vector<std::vector<int>> is_lep = mtx -> get<std::vector<std::vector<int>>>(graph_enum::data_node, "is_lep");
    std::vector<std::vector<int>> is_bq  = mtx -> get<std::vector<std::vector<int>>>(graph_enum::data_node, "is_b");
    for (size_t x(0); x < ptx.size(); ++x){
        if (is_lep[x][0] > 0){ptx[x] -> pdgid = 11;}
        if ( is_bq[x][0] > 0){ptx[x] -> pdgid = 5;}
    }
    return ptx;
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
    auto get_var =[this](
            metric_t* mtx, std::vector<particle_template*>* ptr, auto* rt, particle_enum val
    ) -> void{
        typename std::decay<decltype(rt->training)>::type _vl = {}; 
        for (size_t x(0); x < ptr -> size(); ++x){
            int passed = 1; 
            particle_template* ptr_ = ptr -> at(x); 
            std::map<std::string, particle_template*> ch = ptr_ -> children; 
            if (ch.size() > 0){
                std::vector<particle_template*> ch_ = this -> vectorize(&ch); 
                int b(0), l(0), n(0); 
                for (size_t y(0); y < ch_.size(); ++y){
                    if (ch_[y] -> is_b){b += 1;}
                    else if (ch_[y] -> is_lep){l += 1;}
                    else {n += 1;}
                } 
                if      (l > 0  &&  b > 0 && ch_.size() > 2){passed = +2;} // leptonic
                else if (b >= 1 && n <= 2 && ch_.size() < 5){passed = -2;} // boosted tops merging
                passed = (1000 * n + 100 * l + 10 * b + std::abs(passed)) * sgn(passed); 
            }
            else {passed = ( 100 * int(ptr_ -> is_lep) + 10 * int(ptr_ -> is_b) );}

            switch(val){
                case particle_enum::pt:     _vl.push_back(ptr -> at(x) -> pt );  break; 
                case particle_enum::eta:    _vl.push_back(ptr -> at(x) -> eta);  break; 
                case particle_enum::phi:    _vl.push_back(ptr -> at(x) -> phi);  break; 
                case particle_enum::mass:   _vl.push_back(ptr -> at(x) -> mass); break; 
                case particle_enum::energy: _vl.push_back(ptr -> at(x) -> e  );  break; 
                case particle_enum::is_lep: _vl.push_back(passed); break; 
                default: break;
            }
        }
        rt -> write(this, mtx, &_vl);  
    }; 

    auto maxv =[](std::vector<float>* acx) -> int {
        int idx = 0; 
        float v = acx -> at(0); 
        for (size_t x(0); x < acx -> size(); ++x){
            if (acx -> at(x) < v){continue;}
            v = acx -> at(x); idx = x; 
        }
        return idx; 
    }; 

    auto edge_acc =[](std::vector<int>* pred, std::vector<int>* truth) -> float {
        if (!pred->size()) { return 0.0; }
        float t_pos = 0, p_pos = 0;
        float t_neg = 0, p_neg = 0;
        for (size_t i(0); i < pred -> size(); ++i) {
            t_pos += truth -> at(i) == 1; p_pos += pred -> at(i) == 1; 
            t_neg += truth -> at(i) == 0; p_neg += pred -> at(i) == 0; 
        }
        float acc_pos = (t_pos > 0) ? (p_pos / t_pos) : 0.0;
        float acc_neg = (t_neg > 0) ? (p_neg / t_neg) : 0.0;
        if (t_pos > 0 && t_neg > 0){return (acc_pos + acc_neg) / 2.0;}
        if (t_pos > 0){return acc_pos;}
        return acc_neg;
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
       
        get_var(mtx, &evnt.ptx, &this -> particles_pt, particle_enum::pt); 
        get_var(mtx, &evnt.ptx, &this -> particles_eta, particle_enum::eta); 
        get_var(mtx, &evnt.ptx, &this -> particles_phi, particle_enum::phi); 
        get_var(mtx, &evnt.ptx, &this -> particles_energy, particle_enum::energy); 
        get_var(mtx, &evnt.ptx, &this -> particles_chn, particle_enum::is_lep); 
                       
        get_var(mtx, &evnt.reco_tops_pr, &this -> tops_pr_pt, particle_enum::pt); 
        get_var(mtx, &evnt.reco_tops_pr, &this -> tops_pr_eta, particle_enum::eta); 
        get_var(mtx, &evnt.reco_tops_pr, &this -> tops_pr_phi, particle_enum::phi); 
        get_var(mtx, &evnt.reco_tops_pr, &this -> tops_pr_mass, particle_enum::mass); 
        get_var(mtx, &evnt.reco_tops_pr, &this -> tops_pr_chn, particle_enum::is_lep); 
        this -> tops_pr_scr.write(this, mtx, &evnt.reco_scores_pr);
                       
        get_var(mtx, &evnt.reco_tops_upr, &this -> tops_upr_pt, particle_enum::pt); 
        get_var(mtx, &evnt.reco_tops_upr, &this -> tops_upr_eta, particle_enum::eta); 
        get_var(mtx, &evnt.reco_tops_upr, &this -> tops_upr_phi, particle_enum::phi); 
        get_var(mtx, &evnt.reco_tops_upr, &this -> tops_upr_mass, particle_enum::mass); 
        get_var(mtx, &evnt.reco_tops_upr, &this -> tops_upr_chn, particle_enum::is_lep); 
        this -> tops_upr_scr.write(this, mtx, &evnt.reco_scores_upr);
                       
        get_var(mtx, &evnt.nominal_tops, &this -> tops_nom_pt, particle_enum::pt); 
        get_var(mtx, &evnt.nominal_tops, &this -> tops_nom_eta, particle_enum::eta); 
        get_var(mtx, &evnt.nominal_tops, &this -> tops_nom_phi, particle_enum::phi); 
        get_var(mtx, &evnt.nominal_tops, &this -> tops_nom_mass, particle_enum::mass); 
        get_var(mtx, &evnt.nominal_tops, &this -> tops_nom_chn, particle_enum::is_lep); 
                       
        get_var(mtx, &evnt.truth_tops, &this -> tops_tru_pt, particle_enum::pt); 
        get_var(mtx, &evnt.truth_tops, &this -> tops_tru_eta, particle_enum::eta); 
        get_var(mtx, &evnt.truth_tops, &this -> tops_tru_phi, particle_enum::phi); 
        get_var(mtx, &evnt.truth_tops, &this -> tops_tru_mass, particle_enum::mass); 
        get_var(mtx, &evnt.truth_tops, &this -> tops_tru_chn, particle_enum::is_lep); 

        float avg_edge = edge_acc(&evnt.top_edge_pred, &evnt.top_edge_truth);

        this -> ntops_prd.write(this, mtx, &evnt.n_tops_pred); 
        this -> ntops_tru.write(this, mtx, &evnt.n_tops_truth); 
        this -> proc_idx.write(this, mtx, &evnt.process_ix); 
        this -> edge_prd.write(this, mtx, &avg_edge); 
        this -> ntops_scores.write(this, mtx, &evnt.n_tops_score, true); 
    }
}



