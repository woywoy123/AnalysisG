#ifndef AVERAGE_METRIC_PR_H
#define AVERAGE_METRIC_PR_H

enum process_t : int {
    t_tchan, 
    t_schan, 
    tW,
    ttbar, 
    tt_l, 
    tt_ll, 
    tttt_SM, 
    tttt_m400, 
    tttt_m500, 
    tttt_m600, 
    tttt_m700,
    tttt_m800, 
    tttt_m900, 
    tttt_m1000,
    Z_ll, 
    W_lv,
    ZZ_qqll, 
    WZ_qqll,
    ttH, 
    ttZ_qq,
    ttZ_vv, 
    ttW, 
    ZH, 
    WH,
    llll, 
    lllv, 
    llvv, 
    lvvv,
    invalid
};


namespace processtype {
    namespace t {
        constexpr process_t tchannel = process_t::t_tchan;
        constexpr process_t schannel = process_t::t_schan;
    }; 
    namespace ttbar {
        constexpr process_t inclusive = process_t::ttbar;
        constexpr process_t ll        = process_t::tt_ll;
        constexpr process_t l         = process_t::tt_l;
    }; 
    namespace tttt {
        constexpr process_t SM    = process_t::tttt_SM;
        constexpr process_t m400  = process_t::tttt_m400;
        constexpr process_t m500  = process_t::tttt_m500;
        constexpr process_t m600  = process_t::tttt_m600;
        constexpr process_t m700  = process_t::tttt_m700;
        constexpr process_t m800  = process_t::tttt_m800;
        constexpr process_t m900  = process_t::tttt_m900;
        constexpr process_t m1000 = process_t::tttt_m1000;
    }; 
    namespace Z {
        constexpr process_t ll = process_t::Z_ll;
    }; 
    namespace W {
        constexpr process_t lv = process_t::W_lv;
    }; 
    namespace ZZ {
        constexpr process_t qqll = process_t::ZZ_qqll;
    }; 
    namespace WZ {
        constexpr process_t qqll = process_t::WZ_qqll;
    }; 
    namespace ttZ {
        constexpr process_t qq = process_t::ttZ_qq;
        constexpr process_t vv = process_t::ttZ_vv; 
    }; 
    
    constexpr process_t tW   = process_t::tW;
    constexpr process_t ttW  = process_t::ttW;
    constexpr process_t ttH  = process_t::ttH;
    constexpr process_t ZH   = process_t::ZH;
    constexpr process_t WH   = process_t::WH;
    constexpr process_t llll = process_t::llll;
    constexpr process_t lllv = process_t::lllv; 
    constexpr process_t llvv = process_t::llvv; 
    constexpr process_t lvvv = process_t::lvvv; 
}; 

process_t process_sample(std::string* name);
process_t process_sample(process_t prc);

struct event_idx {
    std::vector<std::vector<int>>   edge_index     = {}; 
    std::vector<std::vector<float>> top_edge_score = {}; 
    std::vector<int>                top_edge_truth = {}; 
    std::vector<int>                top_edge_pred  = {}; 
    std::vector<float>              n_tops_score   = {}; 

    int n_tops_truth = -1; 
    int n_tops_pred  = -1; 
    int process_ix = -1; 
    std::string* file = nullptr;
    
    std::vector<particle_template*> ptx; 
    std::vector<particle_template*> reco_tops_pr; 
    std::vector<float> reco_scores_pr; 

    std::vector<particle_template*> reco_tops_upr;
    std::vector<float> reco_scores_upr; 

    std::vector<particle_template*> nominal_tops; 
    std::vector<particle_template*> truth_tops; 

}; 







#endif
