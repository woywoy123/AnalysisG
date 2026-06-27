#ifndef EVENTS_GNN_EVENT_UTIL_H
#define EVENTS_GNN_EVENT_UTIL_H
#include <templates/event_template.h>
#include "inference/gnn-particles.h"

template <typename g>
static void reduce(element_t* el, std::string key, g* out){
    std::vector<std::vector<g>> tmp;
    if (!el -> get(key, &tmp)){return;}
    (*out) = tmp[0][0]; 
};

template <typename g>
static void reduce(element_t* el, std::string key, std::vector<g>* out, int dim){
    std::vector<std::vector<g>> tmp;
    if (!el -> get(key, &tmp)){return;}
    if (dim == -1){(*out) = tmp[0]; return;}
    for (size_t x(0); x < tmp.size(); ++x){(*out).push_back(tmp[x][0]);}
}; 

template <typename g>
static void print(std::vector<std::vector<g>>* data){
    std::cout << "[" << std::endl; 
    std::cout << "["; 
    for (size_t x(0); x < data -> size(); ++x){
        std::cout << (*data)[x][0]; 
        if (x == data -> size()-1){continue;}
        std::cout << ", "; 
    }
    std::cout << "]," << std::endl; 
    std::cout << "["; 
    for (size_t x(0); x < data -> size(); ++x){
        std::cout << (*data)[x][1]; 
        if (x == data -> size()-1){continue;}
        std::cout << ", "; 
    }
    std::cout << "]" << std::endl; 
    std::cout << "]" << std::endl; 
}; 

template <typename g, typename k>
static void print(std::map<g, std::map<g, k>>* data){
    std::cout << "[" << std::endl;
    int sx = data -> size(); 
    for (int x(0); x < sx; ++x){
        std::cout << "["; 
        for (int y(0); y < sx; ++y){
            std::cout << (*data)[x][y];
            if (y == sx-1){continue;}
            std::cout << ", "; 
        }
        std::cout << "]"; 
        if (x == sx-1){continue;}
        std::cout << "," << std::endl;
    }
    std::cout << "" << std::endl;
    std::cout << "]" << std::endl; 
}; 

template <typename g, typename k>
static void print(std::map<g, k>* data){
    std::cout << "[";
    int sx = data -> size(); 
    for (int x(0); x < sx; ++x){
        std::cout << (*data)[x];
        if (x == sx-1){continue;}
        std::cout << ", "; 
    }
    std::cout << "]" << std::endl; 
}; 


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
    namespace tt {
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
}










#endif
