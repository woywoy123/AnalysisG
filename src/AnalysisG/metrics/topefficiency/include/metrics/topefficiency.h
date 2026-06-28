#ifndef TOPEFFICIENCY_METRIC_H
#define TOPEFFICIENCY_METRIC_H

#include <templates/metric_template.h>

enum class process_t {
    t_tchan, t_schan, tW,
    ttbar, tt_l, tt_ll, tttt_SM, 
    tttt_m400, tttt_m500, tttt_m600, tttt_m700,
    tttt_m800, tttt_m900, tttt_m1000,
    Z_ll, W_lv,
    ZZ_qqll, WZ_qqll,
    ttH, ttZ_qq,
    ttZ_vv, ttW, ZH, WH,
    llll, lllv, llvv, lvvv,
    invalid
};

enum class pagerank_e {
    truth,
    nominal,
    masked,
    unmasked, 
    bias_masked,
    bias_unmasked
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


class particle {
    
    public:
        particle();
        particle(double _pt, double _eta, double _phi, double _mass); 

        ~particle();
        double pt    = 0; 
        double eta   = 0; 
        double phi   = 0; 
        double mass  = 0; 
        double chi2  = 0; 
        double PR    = 0; 
        int leptonic = 0; 
}; 


class Event : public tools {
    public:
        Event();
        ~Event();

        void make_particle(std::map<std::string, std::vector<double>>* vals); 
        process_t process_sample(std::string set); 

        long idx = 0; 
        long epoch = 0;
        long kfold = 0;

        double weight = 0; 
        double msk_recovery  = 0; 
        double nom_recovery  = 0;
        double umsk_recovery = 0; 

        std::vector<double> nom_error  = {};  
        std::vector<double> msk_error  = {}; 
        std::vector<double> umsk_error = {};  

        std::vector<double> top_truth_mass    = {}; 
        std::vector<double> top_nominal_mass  = {}; 
        std::vector<double> top_masked_mass   = {}; 
        std::vector<double> top_unmasked_mass = {}; 

        std::vector<double> top_masked_vPR   = {}; 
        std::vector<double> top_unmasked_vPR = {}; 

        process_t production; 

        std::string modelname = ""; 
        std::string dset_name = "";

        std::vector<particle> top_truth       = {};
        std::vector<particle> top_nominal     = {}; 
        std::vector<particle> top_masked_PR   = {}; 
        std::vector<particle> top_unmasked_PR = {};

}; 

struct particle_data_t {
    // n-Top Clustered 
    std::vector<double>  nominal_kfolds_ntop; 
    std::vector<double>   masked_kfolds_ntop; 
    std::vector<double> unmasked_kfolds_ntop; 

    std::vector<double>  nominal_kfolds_chi2; 
    std::vector<double>   masked_kfolds_chi2; 
    std::vector<double> unmasked_kfolds_chi2; 


    std::vector<double>    truth_kfolds_top_mass; 
    std::vector<double>  nominal_kfolds_top_mass; 
    std::vector<double>   masked_kfolds_top_mass; 
    std::vector<double> unmasked_kfolds_top_mass; 

    std::vector<double>   masked_kfolds_PR; 
    std::vector<double> unmasked_kfolds_PR; 

    std::vector<Event*>  kfolds = {}; 
}; 

struct epoch_t {
    long epoch = -1; 
    std::map<std::string, particle_data_t> evn; 
}; 

class topefficiency_metric: public metric_template
{
    public:
        topefficiency_metric(); 
        ~topefficiency_metric() override; 
        topefficiency_metric* clone() override; 

        std::string to_string(process_t p);
        void add_event(Event* ev); 
        void finalize(); 

        std::vector<Event*> evnts; 
        std::map<   process_t, std::map<long, epoch_t> > proc_evn; 
        std::map< std::string, std::map<long, epoch_t> > generic_data; 

}; 

#endif
