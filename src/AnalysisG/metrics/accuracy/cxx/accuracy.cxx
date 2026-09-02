#include <metrics/accuracy.h>

accuracy_metric::~accuracy_metric(){}
accuracy_metric* accuracy_metric::clone(){return new accuracy_metric();}
accuracy_metric::accuracy_metric(){this -> name = "accuracy";}

void accuracy_metric::start(metric_t* mtx){
    this -> filename = this -> output_path + "/" + mtx -> mode() + ".root"; 
    this -> register_output(this -> dsids_idx.tr_name + "_" + mtx -> mode(), &this -> dsids_idx); 
}

void accuracy_metric::define_variables(metric_t* mtx){
    if (!this -> output_path.size()){
        this -> output_path  = "ProjectName/metrics/accuracy/";
        this -> output_path += mtx -> run_name;
        this -> create_path(this -> output_path);
    }
    this -> start(mtx); 

}


void accuracy_metric::define_metric(metric_t* mtx){

    std::vector<std::vector<int>>   edge_ix  = this -> edge_index(mtx); 
    std::vector<std::vector<float>> edge_sc  = this -> top_edge_score(mtx); 
    std::vector<std::vector<int>>   edge_tr  = this -> top_edge_truth(mtx); 

    std::vector<std::vector<int>>   ntops_tr = this -> ntops_truth(mtx); 
    std::vector<std::vector<float>> ntops_sc = this -> ntops_score(mtx);
    std::vector<particle_template*> ptx      = this -> build_particles(mtx); 

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
            evnt.top_edge_pred.push_back(get_maxidx(&edge_sc[y])); 
        }
        for (size_t y(0); y < btch_ix.size(); ++y){
            if (evnt_ix[x] != btch_ix[y]){continue;}
            evnt.ptx.push_back(ptx[y]); 
            num_nodes += 1;
        }
        
        evnt.file = mtx -> get_filename(x); 
        evnt.n_tops_truth = ntops_tr[x][0]; 
        evnt.n_tops_score = ntops_sc[x]; 
        evnt.n_tops_pred  = get_maxidx(&evnt.n_tops_score); 
        evnt.process_ix   = int(process_sample(evnt.file, this -> dsids_idx.get(mtx -> _mode))); 
        this -> pagerank(&evnt); 
      
        this -> dsids_idx.write(this, mtx, this -> dsids_idx.get(mtx -> _mode)); 
        this -> write_var(mtx, &evnt.ptx, &this -> particles_pt,     particle_enum::pt); 
        this -> write_var(mtx, &evnt.ptx, &this -> particles_eta,    particle_enum::eta); 
        this -> write_var(mtx, &evnt.ptx, &this -> particles_phi,    particle_enum::phi); 
        this -> write_var(mtx, &evnt.ptx, &this -> particles_energy, particle_enum::energy); 
        this -> write_var(mtx, &evnt.ptx, &this -> particles_chn,    particle_enum::is_lep); 
        
        this -> write_var(mtx, &evnt.reco_tops_pr, &this -> tops_pr_pt,   particle_enum::pt); 
        this -> write_var(mtx, &evnt.reco_tops_pr, &this -> tops_pr_eta,  particle_enum::eta); 
        this -> write_var(mtx, &evnt.reco_tops_pr, &this -> tops_pr_phi,  particle_enum::phi); 
        this -> write_var(mtx, &evnt.reco_tops_pr, &this -> tops_pr_mass, particle_enum::mass); 
        this -> write_var(mtx, &evnt.reco_tops_pr, &this -> tops_pr_chn,  particle_enum::is_lep); 
        this -> tops_pr_scr.write(this, mtx, &evnt.reco_scores_pr);
                       
        this -> write_var(mtx, &evnt.reco_tops_upr, &this -> tops_upr_pt,   particle_enum::pt); 
        this -> write_var(mtx, &evnt.reco_tops_upr, &this -> tops_upr_eta,  particle_enum::eta); 
        this -> write_var(mtx, &evnt.reco_tops_upr, &this -> tops_upr_phi,  particle_enum::phi); 
        this -> write_var(mtx, &evnt.reco_tops_upr, &this -> tops_upr_mass, particle_enum::mass); 
        this -> write_var(mtx, &evnt.reco_tops_upr, &this -> tops_upr_chn,  particle_enum::is_lep); 
        this -> tops_upr_scr.write(this, mtx, &evnt.reco_scores_upr);
                       
        this -> write_var(mtx, &evnt.nominal_tops, &this -> tops_nom_pt,   particle_enum::pt); 
        this -> write_var(mtx, &evnt.nominal_tops, &this -> tops_nom_eta,  particle_enum::eta); 
        this -> write_var(mtx, &evnt.nominal_tops, &this -> tops_nom_phi,  particle_enum::phi); 
        this -> write_var(mtx, &evnt.nominal_tops, &this -> tops_nom_mass, particle_enum::mass); 
        this -> write_var(mtx, &evnt.nominal_tops, &this -> tops_nom_chn,  particle_enum::is_lep); 
        this -> tops_nom_scr.write(this, mtx, &evnt.reco_scores_nom);
                      
        this -> write_var(mtx, &evnt.truth_tops, &this -> tops_tru_pt,   particle_enum::pt); 
        this -> write_var(mtx, &evnt.truth_tops, &this -> tops_tru_eta,  particle_enum::eta); 
        this -> write_var(mtx, &evnt.truth_tops, &this -> tops_tru_phi,  particle_enum::phi); 
        this -> write_var(mtx, &evnt.truth_tops, &this -> tops_tru_mass, particle_enum::mass); 
        this -> write_var(mtx, &evnt.truth_tops, &this -> tops_tru_chn,  particle_enum::is_lep); 
        float avg_edge = edge_f1(&evnt.top_edge_pred, &evnt.top_edge_truth);

        this -> ntops_prd.write(this,    mtx, &evnt.n_tops_pred); 
        this -> ntops_tru.write(this,    mtx, &evnt.n_tops_truth); 
        this -> proc_idx.write(this,     mtx, &evnt.process_ix); 
        this -> edge_prd.write(this,     mtx, &avg_edge); 
        this -> ntops_scores.write(this, mtx, &evnt.n_tops_score, true); 
    }
}



