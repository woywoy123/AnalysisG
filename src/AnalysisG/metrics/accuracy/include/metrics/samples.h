#ifndef AVERAGE_METRIC_MAP_H
#define AVERAGE_METRIC_MAP_H
#include <tools/tools.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>

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
    }
    namespace ttbar {
        constexpr process_t inclusive = process_t::ttbar;
        constexpr process_t ll        = process_t::tt_ll;
        constexpr process_t l         = process_t::tt_l;
    } 
    namespace tttt {
        constexpr process_t SM    = process_t::tttt_SM;
        constexpr process_t m400  = process_t::tttt_m400;
        constexpr process_t m500  = process_t::tttt_m500;
        constexpr process_t m600  = process_t::tttt_m600;
        constexpr process_t m700  = process_t::tttt_m700;
        constexpr process_t m800  = process_t::tttt_m800;
        constexpr process_t m900  = process_t::tttt_m900;
        constexpr process_t m1000 = process_t::tttt_m1000;
    } 
    namespace Z {
        constexpr process_t ll = process_t::Z_ll;
    } 
    namespace W {
        constexpr process_t lv = process_t::W_lv;
    } 

    namespace ZZ {
        constexpr process_t qqll = process_t::ZZ_qqll;
    } 
    
    namespace WZ {
        constexpr process_t qqll = process_t::WZ_qqll;
    } 
    namespace ttZ {
        constexpr process_t qq = process_t::ttZ_qq;
        constexpr process_t vv = process_t::ttZ_vv; 
    }
   
    constexpr process_t tW   = process_t::tW;
    constexpr process_t ttW  = process_t::ttW;
    constexpr process_t ttH  = process_t::ttH;
    constexpr process_t ZH   = process_t::ZH;
    constexpr process_t WH   = process_t::WH;
    constexpr process_t llll = process_t::llll;
    constexpr process_t lllv = process_t::lllv; 
    constexpr process_t llvv = process_t::llvv; 
    constexpr process_t lvvv = process_t::lvvv; 
}

struct cdata_t {
    int kfold = -1;
    std::vector<int> ntops_truth = {}; 
    std::vector<std::vector<double>> ntop_score = {};
    std::map<int, std::vector<double>> ntop_edge_accuracy = {};
    std::map<int, std::map<int, std::vector<double>>> ntru_npred_matrix = {}; 
}; 

struct cmodel_t {
    std::map<int, std::map<int, cdata_t>> evaluation_kfold_data = {}; 
    std::map<int, std::map<int, cdata_t>> validation_kfold_data = {}; 
    std::map<int, std::map<int, cdata_t>> training_kfold_data   = {}; 
}; 


class collector: public tools 
{
    public:
        collector(); 
        ~collector();

        cdata_t* get_mode(std::string model, std::string mode, int epoch, int kfold); 
        void add_ntop_truth(std::string mode, std::string model, int epoch, int kfold, int data);
        void add_ntop_edge_accuracy(std::string mode, std::string model, int epoch, int kfold, int ntops, double data);
        void add_ntop_scores(std::string mode, std::string model, int epoch, int kfold, std::vector<double>* data);
        void add_ntru_ntop_scores(std::string mode, std::string model, int epoch, int kfold, int ntru, int ntop, double data);
        std::map<std::string, std::vector<cdata_t*>> get_plts(); 
        std::vector<std::string> model_names = {}; 
        std::vector<std::string> modes = {}; 
        std::vector<int> epochs = {}; 
        std::vector<int> kfolds = {}; 

        std::map<std::string, cmodel_t> model_data = {}; 
};

process_t process_sample(std::string* name, int* dsids = nullptr);
process_t process_sample(process_t prc);

#endif
