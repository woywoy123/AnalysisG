#include <templates/particle_template.h>
#include <metrics/pagerank.h>
#include <metrics/accuracy.h>

std::string mapping(std::string name, collector* cl){
    if (cl -> has_string(&name, "_singletop_" )){return "$t$"                      ;}
    if (cl -> has_string(&name, "_tchan_"     )){return "$t$"                      ;}
    if (cl -> has_string(&name, "_ttbarHT1k_" )){return "$t\\bar{t}$"              ;}
    if (cl -> has_string(&name, "_ttbar_"     )){return "$t\\bar{t}$"              ;}
    if (cl -> has_string(&name, "_ttbarHT1k5_")){return "$t\\bar{t}$"              ;}
    if (cl -> has_string(&name, "_ttbarHT6c_" )){return "$t\\bar{t}$"              ;}
    if (cl -> has_string(&name, "_tt_"        )){return "$t\\bar{t}$"              ;}
    if (cl -> has_string(&name, "_ttee."      )){return "$t\\bar{t}\\ell\\ell$"    ;}
    if (cl -> has_string(&name, "_ttmumu."    )){return "$t\\bar{t}\\ell\\ell$"    ;}
    if (cl -> has_string(&name, "_tttautau."  )){return "$t\\bar{t}\\ell\\ell$"    ;}
    if (cl -> has_string(&name, "_ttW."       )){return "$t\\bar{t}V$"             ;}
    if (cl -> has_string(&name, "_ttZnunu."   )){return "$t\\bar{t}V$"             ;}
    if (cl -> has_string(&name, "_ttZqq."     )){return "$t\\bar{t}V$"             ;}
    if (cl -> has_string(&name, "_ttH125_"    )){return "$t\\bar{t}H$"             ;}
    if (cl -> has_string(&name, "_Wt_"        )){return "$Wt$"                     ;}
    if (cl -> has_string(&name, "_tW."        )){return "$tV$"                     ;}
    if (cl -> has_string(&name, "_tW_"        )){return "$tV$"                     ;}
    if (cl -> has_string(&name, "_tZ."        )){return "$tV$"                     ;}
    if (cl -> has_string(&name, "_SM4topsNLO" )){return "$t\\bar{t}t\\bar{t}$"     ;}
    if (cl -> has_string(&name, "_WlvZqq"     )){return "$WZ$"                     ;}
    if (cl -> has_string(&name, "_WqqZll"     )){return "$WZ$"                     ;}
    if (cl -> has_string(&name, "_WqqZvv"     )){return "$WZ$"                     ;}
    if (cl -> has_string(&name, "_WplvWmqq"   )){return "$WW$"                     ;}
    if (cl -> has_string(&name, "_WpqqWmlv"   )){return "$WW$"                     ;}
    if (cl -> has_string(&name, "_ZqqZll"     )){return "$ZZ$"                     ;}
    if (cl -> has_string(&name, "_ZqqZvv"     )){return "$ZZ$"                     ;}
    if (cl -> has_string(&name, "_WH125."     )){return "$VH$"                     ;}
    if (cl -> has_string(&name, "_ZH125_"     )){return "$VH$"                     ;}
    if (cl -> has_string(&name, "_WH125_"     )){return "$VH$"                     ;}
    if (cl -> has_string(&name, "_Wenu_"      )){return "$V\\ell\\nu$"             ;}
    if (cl -> has_string(&name, "_Wmunu_"     )){return "$V\\ell\\nu$"             ;}
    if (cl -> has_string(&name, "_Wtaunu_"    )){return "$V\\ell\\nu$"             ;}
    if (cl -> has_string(&name, "_Zee_"       )){return "$V\\ell\\ell$"            ;}
    if (cl -> has_string(&name, "_Zmumu_"     )){return "$V\\ell\\ell$"            ;}
    if (cl -> has_string(&name, "_Ztautau_"   )){return "$V\\ell\\ell$"            ;}
    if (cl -> has_string(&name, "_llll"       )){return "$\\ell\\ell\\ell\\ell$"   ;}
    if (cl -> has_string(&name, "_lllv"       )){return "$\\ell\\ell\\ell\\nu$"    ;}
    if (cl -> has_string(&name, "_llvv"       )){return "$\\ell\\ell\\nu\\nu$"     ;}
    if (cl -> has_string(&name, "_lvvv"       )){return "$\\ell\\nu\\nu\\nu$"      ;}
    return "undef"; 
}

process_t process_sample(std::string* name) {
    size_t start_pos = name -> find("mc16");
    if (start_pos != std::string::npos){start_pos = name -> find('.', start_pos) + 1;}
    else {start_pos = 0;}
    
    size_t end_pos = name -> find('.', start_pos);
    if (end_pos == std::string::npos || start_pos >= name -> length()) {return process_t::invalid;}

    int dsid = -1;
    try {dsid = std::stoi(name -> substr(start_pos, end_pos - start_pos));} 
    catch (...) {return process_t::invalid;}

    static const std::unordered_map<int, process_t> dsid_map = {
        // --- 4 Tops ---
        {312440, processtype::tttt::m400},
        {312441, processtype::tttt::m500},
        {312442, processtype::tttt::m600},
        {312443, processtype::tttt::m700},
        {312444, processtype::tttt::m800},
        {312445, processtype::tttt::m900},
        {312446, processtype::tttt::m1000},
        {412043, processtype::tttt::SM},

        // --- Higgs & Boson Associated ---
        {342284, processtype::WH},
        {342285, processtype::ZH},
        {346344, processtype::ttH}, // semilep 
        {346345, processtype::ttH}, // dilep

        // --- Diboson ---
        {363356, processtype::ZZ::qqll},
        {363358, processtype::WZ::qqll},

        // --- Z+Jets (Zll) ---
        {364100, processtype::Z::ll}, {364101, processtype::Z::ll}, {364102, processtype::Z::ll},
        {364103, processtype::Z::ll}, {364104, processtype::Z::ll}, {364105, processtype::Z::ll},
        {364106, processtype::Z::ll}, {364107, processtype::Z::ll}, {364108, processtype::Z::ll},
        {364109, processtype::Z::ll}, {364110, processtype::Z::ll}, {364111, processtype::Z::ll},
        {364112, processtype::Z::ll}, {364113, processtype::Z::ll}, {364114, processtype::Z::ll},
        {364115, processtype::Z::ll}, {364116, processtype::Z::ll}, {364117, processtype::Z::ll},
        {364118, processtype::Z::ll}, {364119, processtype::Z::ll}, {364120, processtype::Z::ll},
        {364121, processtype::Z::ll}, {364122, processtype::Z::ll}, {364123, processtype::Z::ll},
        {364124, processtype::Z::ll}, {364125, processtype::Z::ll}, {364126, processtype::Z::ll},
        {364127, processtype::Z::ll}, {364133, processtype::Z::ll}, {364135, processtype::Z::ll},
        {364136, processtype::Z::ll}, {364137, processtype::Z::ll}, {364138, processtype::Z::ll},
        {364139, processtype::Z::ll}, {364140, processtype::Z::ll}, {364141, processtype::Z::ll},

        // --- W+Jets (Wlnu) ---
        {364165, processtype::W::lv}, {364166, processtype::W::lv}, {364167, processtype::W::lv},
        {364168, processtype::W::lv}, {364169, processtype::W::lv}, {364181, processtype::W::lv},
        {364182, processtype::W::lv}, {364183, processtype::W::lv}, {364197, processtype::W::lv},

        // --- Multi-Lepton ---
        {364250, processtype::llll},
        {364253, processtype::lllv},
        {364254, processtype::llvv},

        // --- Top Pairs (Inclusive / Sliced) ---
        {407342, processtype::ttbar::inclusive}, {407343, processtype::ttbar::inclusive}, 
        {407344, processtype::ttbar::inclusive}, {407348, processtype::ttbar::inclusive}, 
        {407349, processtype::ttbar::inclusive}, {407350, processtype::ttbar::inclusive},
        {410470, processtype::ttbar::inclusive}, {411073, processtype::ttbar::inclusive},
        {411074, processtype::ttbar::inclusive}, {411075, processtype::ttbar::inclusive},
        {411082, processtype::ttbar::inclusive}, {412066, processtype::ttbar::inclusive},
        {412067, processtype::ttbar::inclusive}, {412068, processtype::ttbar::inclusive},

        // --- Top Associated (V) ---
        {410155, processtype::ttW},
        {410156, processtype::ttZ::vv}, // nunu
        {410157, processtype::ttZ::qq},

        // --- Top Pairs (Specific Decays) ---
        {410218, processtype::ttbar::ll}, {410219, processtype::ttbar::ll}, {410220, processtype::ttbar::ll},
        {410464, processtype::ttbar::l},  {410465, processtype::ttbar::ll}, {410472, processtype::ttbar::ll},
        {410480, processtype::ttbar::l},  {410482, processtype::ttbar::ll}, {410557, processtype::ttbar::l},
        {410558, processtype::ttbar::ll}, {411076, processtype::ttbar::ll}, {411077, processtype::ttbar::ll},
        {411078, processtype::ttbar::ll}, {411085, processtype::ttbar::ll}, {411086, processtype::ttbar::ll},
        {411087, processtype::ttbar::ll}, {412069, processtype::ttbar::ll}, {412070, processtype::ttbar::ll},
        {412071, processtype::ttbar::ll},

        // --- Single Top (tchan, schan, tW) ---
        {410560, processtype::t::tchannel}, // lept
        {410658, processtype::t::tchannel},
        {410659, processtype::t::tchannel},
        {411033, processtype::t::tchannel},
        {412004, processtype::t::tchannel},
        
        {410644, processtype::t::schannel}, 
        {410645, processtype::t::schannel},
        {411034, processtype::t::schannel},
        {411035, processtype::t::schannel},
        
        // Note: Grouping inclusive antitop (Wt) into tW 
        {410646, processtype::tW}, {410647, processtype::tW}, 
        {410654, processtype::tW}, {410655, processtype::tW}, 
        {411036, processtype::tW}, {411037, processtype::tW},
        {412002, processtype::tW}
    };
    auto it = dsid_map.find(dsid);
    if (it != dsid_map.end()){return it->second;}
    return process_t::invalid;
} 


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

void accuracy_metric::pagerank(
        std::map<int, std::map<std::string, particle_template*>>* clust, 
        std::map<std::string, std::vector<particle_template*>>* out,
        std::map<std::string, float>* bin_out,
        std::map<int, std::map<int, float>>* bin_data
){
    int s = 0; 
    int e = clust -> size(); 
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
                if (mps.size()){continue;}
                break; 
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


void accuracy_metric::pagerank(event_idx* mtx){
    std::map<int, std::map<int, float>> bin_top;  
    std::map<int, std::map<int, particle_template*>>         real_tops; 
    std::map<int, std::map<int, particle_template*>>      nominal_tops;
    std::map<int, std::map<std::string, particle_template*>> reco_tops; 

    std::map<int, std::map<int, float>> sc_tops;  
    std::map<int, std::map<std::string, particle_template*>> umsk_tops; 

    for (size_t x(0); x < mtx -> edge_index[0].size(); ++x){
        int src = mtx -> edge_index[0][x]; 
        int dst = mtx -> edge_index[1][x];  

        float top_0 = mtx -> top_edge_score[x][0]; 
        float top_1 = mtx -> top_edge_score[x][1]; 

        particle_template* ptr = mtx -> ptx[dst]; 
        std::string hx = ptr -> hash; 

        umsk_tops[src][hx] = ptr; 
        sc_tops[src][dst] = top_1;

        if (mtx -> top_edge_truth[x]){real_tops[src][dst] = ptr;}
        if (top_0 >= top_1){continue;}

        bin_top[src][dst] = top_1;
        reco_tops[src][hx] = ptr; 
        reco_tops[dst][hx] = ptr;
        nominal_tops[src][dst] = ptr; 
    }

    // -------------- build truth tops --------------- //
    mtx -> truth_tops = this -> build_top(&real_tops); 

    // --------------- build nominal tops ------------------ //
    mtx -> nominal_tops = this -> build_top(&nominal_tops); 

    // --------------- build pageranked tops ------------------ //
    std::map<std::string, float> reco_tops_pagerank = {}; 
    std::map<std::string, std::vector<particle_template*>> reco_tops_pr = {}; 
    this -> pagerank(&reco_tops, &reco_tops_pr, &reco_tops_pagerank, &bin_top); 

    std::map<std::string, float>::iterator itr = reco_tops_pagerank.begin();
    for (; itr != reco_tops_pagerank.end(); ++itr){
        particle_template* px = nullptr; 
        this -> sum(&reco_tops_pr[itr -> first], &px); 
        if (!px){continue;}
        mtx -> reco_tops_pr.push_back(px); 
        mtx -> reco_scores_pr.push_back(reco_tops_pagerank[itr -> first]); 
    }

    // --------------- build unmasked pageranked tops ------------------ //
    std::map<std::string, float> reco_tops_unpagerank = {}; 
    std::map<std::string, std::vector<particle_template*>> reco_tops_upr = {}; 
    this -> pagerank(&umsk_tops, &reco_tops_upr, &reco_tops_unpagerank, &sc_tops); 

    itr = reco_tops_unpagerank.begin();
    for (; itr != reco_tops_unpagerank.end(); ++itr){
        particle_template* px = nullptr; 
        this -> sum(&reco_tops_upr[itr -> first], &px); 
        if (!px){continue;}
        mtx -> reco_tops_upr.push_back(px); 
        mtx -> reco_scores_upr.push_back(reco_tops_unpagerank[itr -> first]); 
    }
}

