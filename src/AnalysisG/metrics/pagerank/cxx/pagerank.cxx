#include <metrics/pagerank.h>
#include <fstream>

pagerank_metric::~pagerank_metric(){}
pagerank_metric* pagerank_metric::clone(){return new pagerank_metric();}
pagerank_metric::pagerank_metric(){this -> name = "pagerank";}

void pagerank_metric::define_variables(){
    std::vector<std::string> modes_ = {"training", "validation", "evaluation"}; 
    for (size_t x(0); x < modes_.size(); ++x){
        std::string key = "event_" + modes_[x]; 
        this -> register_output(key, "top_truth_num_nodes", &this -> top_truth_num_nodes); 
        this -> register_output(key, "top_truth_pt"       , &this -> top_truth_pt ); 
        this -> register_output(key, "top_truth_eta"      , &this -> top_truth_eta); 
        this -> register_output(key, "top_truth_phi"      , &this -> top_truth_phi); 
        this -> register_output(key, "top_truth_energy"   , &this -> top_truth_energy); 

        this -> register_output(key, "top_truth_px"       , &this -> top_truth_px  ); 
        this -> register_output(key, "top_truth_py"       , &this -> top_truth_py  );
        this -> register_output(key, "top_truth_pz"       , &this -> top_truth_pz  ); 
        this -> register_output(key, "top_truth_mass"     , &this -> top_truth_mass);

        this -> register_output(key, "top_pr_reco_num_nodes", &this -> top_pr_reco_num_nodes); 
        this -> register_output(key, "top_pr_reco_pt"       , &this -> top_pr_reco_pt       ); 
        this -> register_output(key, "top_pr_reco_eta"      , &this -> top_pr_reco_eta      );
        this -> register_output(key, "top_pr_reco_phi"      , &this -> top_pr_reco_phi      ); 
        this -> register_output(key, "top_pr_reco_energy"   , &this -> top_pr_reco_energy   ); 

        this -> register_output(key, "top_pr_reco_px"       , &this -> top_pr_reco_px      ); 
        this -> register_output(key, "top_pr_reco_py"       , &this -> top_pr_reco_py      );
        this -> register_output(key, "top_pr_reco_pz"       , &this -> top_pr_reco_pz      ); 
        this -> register_output(key, "top_pr_reco_mass"     , &this -> top_pr_reco_mass    );
        this -> register_output(key, "top_pr_reco_pagerank" , &this -> top_pr_reco_pagerank); 

        this -> register_output(key, "top_nom_reco_num_nodes", &this -> top_nom_reco_num_nodes); 
        this -> register_output(key, "top_nom_reco_pt"       , &this -> top_nom_reco_pt       ); 
        this -> register_output(key, "top_nom_reco_eta"      , &this -> top_nom_reco_eta      );
        this -> register_output(key, "top_nom_reco_phi"      , &this -> top_nom_reco_phi      ); 
        this -> register_output(key, "top_nom_reco_energy"   , &this -> top_nom_reco_energy   ); 

        this -> register_output(key, "top_nom_reco_px"       , &this -> top_nom_reco_px   ); 
        this -> register_output(key, "top_nom_reco_py"       , &this -> top_nom_reco_py   );
        this -> register_output(key, "top_nom_reco_pz"       , &this -> top_nom_reco_pz   ); 
        this -> register_output(key, "top_nom_reco_mass"     , &this -> top_nom_reco_mass );
        this -> register_output(key, "top_nom_reco_score"    , &this -> top_nom_reco_score); 

        this -> register_output(key, "process_mapping"       , &this -> process_mapping); 
    }
}

void pagerank_metric::event(){
    this -> write("event_" + this -> mode, "top_truth_num_nodes"   , &this -> top_truth_num_nodes); 
    this -> write("event_" + this -> mode, "top_truth_pt"          , &this -> top_truth_pt       ); 
    this -> write("event_" + this -> mode, "top_truth_eta"         , &this -> top_truth_eta      ); 
    this -> write("event_" + this -> mode, "top_truth_phi"         , &this -> top_truth_phi      ); 
    this -> write("event_" + this -> mode, "top_truth_energy"      , &this -> top_truth_energy   ); 

    this -> write("event_" + this -> mode, "top_truth_px"          , &this -> top_truth_px  ); 
    this -> write("event_" + this -> mode, "top_truth_py"          , &this -> top_truth_py  );
    this -> write("event_" + this -> mode, "top_truth_pz"          , &this -> top_truth_pz  ); 
    this -> write("event_" + this -> mode, "top_truth_mass"        , &this -> top_truth_mass);

    this -> write("event_" + this -> mode, "top_pr_reco_num_nodes" , &this -> top_pr_reco_num_nodes); 
    this -> write("event_" + this -> mode, "top_pr_reco_pt"        , &this -> top_pr_reco_pt       ); 
    this -> write("event_" + this -> mode, "top_pr_reco_eta"       , &this -> top_pr_reco_eta      );
    this -> write("event_" + this -> mode, "top_pr_reco_phi"       , &this -> top_pr_reco_phi      ); 
    this -> write("event_" + this -> mode, "top_pr_reco_energy"    , &this -> top_pr_reco_energy   ); 

    this -> write("event_" + this -> mode, "top_pr_reco_px"        , &this -> top_pr_reco_px      ); 
    this -> write("event_" + this -> mode, "top_pr_reco_py"        , &this -> top_pr_reco_py      );
    this -> write("event_" + this -> mode, "top_pr_reco_pz"        , &this -> top_pr_reco_pz      ); 
    this -> write("event_" + this -> mode, "top_pr_reco_mass"      , &this -> top_pr_reco_mass    );
    this -> write("event_" + this -> mode, "top_pr_reco_pagerank"  , &this -> top_pr_reco_pagerank); 

    this -> write("event_" + this -> mode, "top_nom_reco_num_nodes", &this -> top_nom_reco_num_nodes); 
    this -> write("event_" + this -> mode, "top_nom_reco_pt"       , &this -> top_nom_reco_pt       ); 
    this -> write("event_" + this -> mode, "top_nom_reco_eta"      , &this -> top_nom_reco_eta      );
    this -> write("event_" + this -> mode, "top_nom_reco_phi"      , &this -> top_nom_reco_phi      ); 
    this -> write("event_" + this -> mode, "top_nom_reco_energy"   , &this -> top_nom_reco_energy   ); 

    this -> write("event_" + this -> mode, "top_nom_reco_px"       , &this -> top_nom_reco_px   ); 
    this -> write("event_" + this -> mode, "top_nom_reco_py"       , &this -> top_nom_reco_py   );
    this -> write("event_" + this -> mode, "top_nom_reco_pz"       , &this -> top_nom_reco_pz   ); 
    this -> write("event_" + this -> mode, "top_nom_reco_mass"     , &this -> top_nom_reco_mass );
    this -> write("event_" + this -> mode, "top_nom_reco_score"    , &this -> top_nom_reco_score); 
    this -> write("event_" + this -> mode, "process_mapping"       , &this -> process_mapping, true); // <---- finalize the event

    this -> top_truth_num_nodes.clear();
    this -> top_truth_pt.clear(); 
    this -> top_truth_eta.clear();
    this -> top_truth_phi.clear(); 
    this -> top_truth_energy.clear(); 

    this -> top_truth_px.clear(); 
    this -> top_truth_py.clear();
    this -> top_truth_pz.clear(); 
    this -> top_truth_mass.clear();

    this -> top_pr_reco_num_nodes.clear(); 
    this -> top_pr_reco_pt.clear(); 
    this -> top_pr_reco_eta.clear();
    this -> top_pr_reco_phi.clear(); 
    this -> top_pr_reco_energy.clear(); 

    this -> top_pr_reco_px.clear(); 
    this -> top_pr_reco_py.clear();
    this -> top_pr_reco_pz.clear(); 
    this -> top_pr_reco_mass.clear();
    this -> top_pr_reco_pagerank.clear(); 

    this -> top_nom_reco_num_nodes.clear(); 
    this -> top_nom_reco_pt.clear(); 
    this -> top_nom_reco_eta.clear();
    this -> top_nom_reco_phi.clear(); 
    this -> top_nom_reco_energy.clear(); 

    this -> top_nom_reco_px.clear(); 
    this -> top_nom_reco_py.clear();
    this -> top_nom_reco_pz.clear(); 
    this -> top_nom_reco_mass.clear();
    this -> top_nom_reco_score.clear(); 

    this -> process_mapping = -1; 
}

void pagerank_metric::batch(){}
void pagerank_metric::end(){
    std::map<std::string, std::map<std::string, long>>::iterator itr; 
    std::string kx = "<file-mapping>\n";
    for (itr = this -> file_maps.begin(); itr != this -> file_maps.end(); ++itr){
        kx += "========" + itr -> first + "========\n"; 
        std::map<std::string, long>::iterator itp = itr -> second.begin(); 
        for (; itp != itr -> second.end(); ++itp){kx += itp -> first + "::" + std::to_string(itp -> second) + "\n";}
    }
    kx += "<file-statistics>\n"; 
    for (itr = this -> file_stats.begin(); itr != this -> file_stats.end(); ++itr){
        kx += "========" + itr -> first + "========\n"; 
        std::map<std::string, long>::iterator itp = itr -> second.begin(); 
        for (; itp != itr -> second.end(); ++itp){kx += itp -> first + "::" + std::to_string(itp -> second) + "\n";}
    }
    std::string out = this -> output_path; 
    std::ofstream files(out + ".txt"); 
    files << kx; 
    files.close(); 
}

std::vector<particle_template*> pagerank_metric::build_top(std::map<int, std::map<int, particle_template*>>* mx){
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
        for (size_t x(0); x < tmp.size(); ++x){px -> register_parent(tmp[x]);}
        tops_[mass] = px;
        out.push_back(px); 
    }
    return out; 
}

void pagerank_metric::pagerank(
        std::map<int, std::map<std::string, particle_template*>>* clust, 
        std::map<std::string, std::vector<particle_template*>>* out,
        std::map<std::string, float>* bin_out,
        std::map<int, std::map<int, float>>* bin_data, 
        int batch_offset
){
    int s = batch_offset; 
    int e = batch_offset + clust -> size(); 
    float n_nodes = 1.0 / float(clust -> size()); 

    std::map<int, std::map<int, float>> Mij; 
    std::map<int, std::map<std::string, particle_template*>>::iterator itr;

    for (itr = clust -> begin(); itr != clust -> end(); ++itr){
        int src = itr -> first; 
        for (int y(s); y < e; ++y){Mij[src][y] = (src != y)*(*bin_data)[src][y];}
    }

    std::map<int, float> pr_;
    for (int y(s); y < e; ++y){
        float sm = 0; 
        for (int x(s); x < e; ++x){sm += Mij[x][y];} 
        sm = ((sm) ? 1.0/sm : 0); 
        for (int x(s); x < e; ++x){Mij[x][y] = ((sm) ? Mij[x][y]*sm : n_nodes) * this -> alpha;}
        pr_[y] = (*bin_data)[y][y] * n_nodes;  
    }

    std::map<int, float> PR = pr_; 
    for (size_t t(0); t < this -> max_itr; ++t){
        pr_.clear(); 
        float sx = 0; 
        for (int src(s); src < e; ++src){
            for (int x(s); x < e; ++x){pr_[src] += (Mij[src][x]*PR[x]);}
            pr_[src] += (1 - this -> alpha) * n_nodes; 
            sx += pr_[src]; 
        }
         
        sx = 1.0 / sx; 
        float norm = 0; 
        for (itr = clust -> begin(); itr != clust -> end(); ++itr){
            pr_[itr -> first] = pr_[itr -> first] * sx;
            norm += std::abs(pr_[itr -> first] - PR[itr -> first]); 
            PR[itr -> first] = pr_[itr -> first]; 
        }

        if (norm > this -> norm_lim){continue;}

        norm = 0; 
        for (int x(s); x < e; ++x){
            float sc = 0; 
            for (int y(s); y < e; ++y){sc += (x != y) * Mij[x][y] * (pr_[y]);}
            PR[x] = sc; norm += sc;
        }
        if (!norm){break;}
        for (int x(s); x < e; ++x){PR[x] = PR[x] / norm;}
        break; 
    }

    for (itr = clust -> begin(); itr != clust -> end(); ++itr){
        if (!PR[itr -> first]){continue;}
        std::map<std::string, particle_template*> tmp; 
        std::map<std::string, particle_template*>::iterator itp;
        for (itp = itr -> second.begin(); itp != itr -> second.end(); ++itp){
            particle_template* ptr = itp -> second; 
            if ((*bin_data)[itr -> first][ptr -> index] < 0.5){continue;}
            tmp[ptr -> hash] = ptr;

            std::map<std::string, particle_template*> mps = (*clust)[ptr -> index]; 
            std::map<std::string, particle_template*>::iterator itx = mps.begin(); 
            for (; itx != mps.end(); ++itx){
                ptr = itx -> second; 
                if (tmp.count(ptr -> hash) || clust -> count(ptr -> index)){continue;}
                tmp[ptr -> hash] = itx -> second;
                mps = (*clust)[ptr -> index]; 
                itx = mps.begin(); 
            }
        }
        if (tmp.size() <= 2){continue;}
        std::string hash = ""; 
        for (itp = tmp.begin(); itp != tmp.end(); ++itp){hash = this -> tools::hash(hash + itp -> first);}
        if (out -> count(hash)){continue;}
        for (itp = tmp.begin(); itp != tmp.end(); ++itp){(*bin_out)[hash] += PR[itp -> second -> index];}
        (*out)[hash] = this -> vectorize(&tmp); 
    }
}


void pagerank_metric::define_metric(metric_t* mtx){
    this -> mode  = mtx -> mode(); 
    this -> kfold = mtx -> kfold; 
    this -> epoch = mtx -> epoch; 

    std::vector<long> batch_idx              = mtx -> get<std::vector<long>>(graph_enum::batch_index, "index"); 
    std::vector<std::vector<int>>  edge_idx  = mtx -> get<std::vector<std::vector<int>>>(graph_enum::edge_index, "index"); 
    std::vector<std::vector<int>> top_edge_t = mtx -> get<std::vector<std::vector<int>>>(graph_enum::truth_edge, "top_edge"); 

    std::vector<std::vector<float>> edge_sc  = mtx -> get<std::vector<std::vector<float>>>( graph_enum::pred_extra, "top_edge_score"); 
    std::vector<std::vector<double>> pt      = mtx -> get<std::vector<std::vector<double>>>(graph_enum::data_node , "pt"); 
    std::vector<std::vector<double>> eta     = mtx -> get<std::vector<std::vector<double>>>(graph_enum::data_node , "eta"); 
    std::vector<std::vector<double>> phi     = mtx -> get<std::vector<std::vector<double>>>(graph_enum::data_node , "phi"); 
    std::vector<std::vector<double>> energy  = mtx -> get<std::vector<std::vector<double>>>(graph_enum::data_node , "energy"); 
    std::vector<particle_template*> ptx = this -> make_particle(&pt, &eta, &phi, &energy); 

    std::map<long, int> node_offsets; 
    std::map<long, std::map<int, std::map<int, float>>> bin_top;  
    std::map<long, std::map<int, std::map<int, particle_template*>>> real_tops; 
    std::map<long, std::map<int, std::map<int, particle_template*>>> nominal_tops; 
    std::map<long, std::map<int, std::map<std::string, particle_template*>>> reco_tops; 

    for (size_t x(0); x < edge_idx[0].size(); ++x){
        int src = edge_idx[0][x]; 
        int dst = edge_idx[1][x];  
        long bx = batch_idx[dst]; 
        node_offsets[bx] += (src == dst); 

        float top_0 = edge_sc[x][0]; 
        float top_1 = edge_sc[x][1]; 
        bin_top[bx][src][dst] = top_1;
        particle_template* ptr = ptx[dst]; 
        if (top_edge_t[x][0]){real_tops[bx][src][dst] = ptr;}
        if (top_0 >= top_1){continue;}
        std::string hx = ptr -> hash; 
        reco_tops[bx][src][hx] = ptr; 
        reco_tops[bx][dst][hx] = ptr;
        nominal_tops[bx][src][dst] = ptr; 
    }

    int offset_node = 0; 
    std::map<long, std::map<int, std::map<int, particle_template*>>>::iterator its = real_tops.begin();
    for (; its != real_tops.end(); ++its){
        // -------------- build truth tops --------------- //
        std::vector<particle_template*> top_truth = this -> build_top(&real_tops[its -> first]); 

        std::string fn = ""; 
        std::string* fname = mtx -> get_filename(its -> first); 
        std::vector<std::string> spl = this -> split(*fname, "/"); 
        if (spl.size() > 2){fn = spl[spl.size()-2] + "::" + spl[spl.size()-1];}
        if (!this -> file_maps[this -> mode].count(fn)){
            this -> file_maps[this -> mode][fn] = this -> file_stats[this -> mode].size();
        }
        this -> process_mapping = this -> file_maps[this -> mode][fn];  
        this -> file_stats[this -> mode][fn]++;

        for (size_t x(0); x < top_truth.size(); ++x){
            particle_template* tt = top_truth[x]; 
            std::map<std::string, particle_template*> px = tt -> parents; 
            this -> top_truth_num_nodes.push_back(px.size()); 
            this -> top_truth_pt.push_back(tt -> pt / 1000.0); 
            this -> top_truth_eta.push_back(tt -> eta); 
            this -> top_truth_phi.push_back(tt -> phi); 
            this -> top_truth_energy.push_back(tt -> e / 1000.0); 

            this -> top_truth_px.push_back(tt -> px / 1000.0); 
            this -> top_truth_py.push_back(tt -> py / 1000.0); 
            this -> top_truth_pz.push_back(tt -> pz / 1000.0); 

            this -> top_truth_mass.push_back(tt -> mass / 1000.0); 
        }

        // --------------- build nominal tops ------------------ //
        std::map<std::string, particle_template*> tops_; 
        std::map<int, std::map<int, particle_template*>>::iterator itr = nominal_tops[its -> first].begin(); 
        for (; itr != nominal_tops[its -> first].end(); ++itr){
            float sc = 0; 
            std::map<std::string, particle_template*> tmp; 
            std::map<int, particle_template*>::iterator itx = nominal_tops[its -> first][itr -> first].begin(); 
            for (; itx != nominal_tops[its -> first][itr -> first].end(); ++itx){
                sc += bin_top[its -> first][itr -> first][itx -> first]; 
                tmp[itx -> second -> hash] = itx -> second;
            }
            std::vector<particle_template*> vc = this -> vectorize(&tmp); 
            if (!vc.size()){continue;}
            particle_template* px = nullptr; 
            this -> sum(&vc, &px); 
            if (tops_.count(px -> hash)){continue;}
            tops_[px -> hash] = px; 

            this -> top_nom_reco_num_nodes.push_back(vc.size()); 
            this -> top_nom_reco_pt.push_back(px -> pt / 1000.0); 
            this -> top_nom_reco_eta.push_back(px -> eta); 
            this -> top_nom_reco_phi.push_back(px -> phi); 
            this -> top_nom_reco_energy.push_back(px -> e / 1000.0); 

            this -> top_nom_reco_px.push_back(px -> px / 1000.0); 
            this -> top_nom_reco_py.push_back(px -> py / 1000.0); 
            this -> top_nom_reco_pz.push_back(px -> pz / 1000.0); 

            this -> top_nom_reco_mass.push_back(px -> mass / 1000.0); 
            this -> top_nom_reco_score.push_back(sc); 
        }

        // -------------- build reco tops --------------- //
        std::map<std::string, float> reco_tops_pagerank = {}; 
        std::map<std::string, std::vector<particle_template*>> reco_tops_pr = {}; 
        this -> pagerank(&reco_tops[its -> first], &reco_tops_pr, &reco_tops_pagerank, &bin_top[its -> first], offset_node); 

        std::map<std::string, float>::iterator itp = reco_tops_pagerank.begin(); 
        for (; itp != reco_tops_pagerank.end(); ++itp){
            std::vector<particle_template*> tv = reco_tops_pr[itp -> first]; 
            particle_template* tr = nullptr; 
            this -> sum(&tv, &tr); 
            
            this -> top_pr_reco_num_nodes.push_back(tv.size()); 
            this -> top_pr_reco_pt.push_back(tr -> pt / 1000.0); 
            this -> top_pr_reco_eta.push_back(tr -> eta); 
            this -> top_pr_reco_phi.push_back(tr -> phi); 
            this -> top_pr_reco_energy.push_back(tr -> e / 1000.0); 

            this -> top_pr_reco_px.push_back(tr -> px / 1000.0); 
            this -> top_pr_reco_py.push_back(tr -> py / 1000.0); 
            this -> top_pr_reco_pz.push_back(tr -> pz / 1000.0); 

            this -> top_pr_reco_mass.push_back(tr -> mass / 1000.0); 
            this -> top_pr_reco_pagerank.push_back(itp -> second); 
        }

        this -> event(); 
        offset_node += node_offsets[its -> first]; 
    }
}

