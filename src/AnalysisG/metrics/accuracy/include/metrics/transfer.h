#ifndef AVERAGE_LAYER_H
#define AVERAGE_LAYER_H
#include <metrics/particle.h>
#include <metrics/samples.h>
#include <tools/tools.h>

class pdata {
    public:
        pdata();
        ~pdata(); 

        std::vector<double> pt; 
        std::vector<double> eta; 
        std::vector<double> phi; 
        std::vector<double> energy; 
        std::vector<double> mass; 
        std::vector<double> score; 
        std::vector<int> chn;

        std::string type = ""; 
        std::string mode = ""; 
}; 


class edata : public tools
{
    public:
        edata();
        ~edata(); 

        std::vector<pdata*> training   = {}; 
        std::vector<pdata*> validation = {}; 
        std::vector<pdata*> evaluation = {}; 

        std::vector<double> ntop_score = {}; 
        int ntop_tru = 0; 
        int ntop_prd = 0; 
        int dsid     = 0; 
        int prc_idx  = 0; 

        double acc_edge  = 0; 

        int kfold = 0;
        int epoch = 0;
        std::string mrk = ""; 
}; 

struct edata_t {
    edata_t()                   = default;
    edata_t(const edata_t&)     = default;
    edata_t(edata_t&&) noexcept = default;

    edata_t& operator=(edata_t&&) noexcept = default;
    edata_t& operator=(const edata_t&)     = default;

    int ntru = 0; 
    int npre = 0; 
    std::vector<double> pred; 
    double acc_edges = 0; 

    process_t prc = process_t::invalid; 
    std::vector<rectop> truth;
    std::vector<rectop> nominal;
    std::vector<rectop> umasked;
    std::vector<rectop> masked; 
   
    int njets  = 0;  
    int nbjets = 0;
    int nleps  = 0; 
}; 


struct evn_t {
    evn_t();

    void mass(double w); 
    void assign(int tru, int pred); 

    void weights(double w);
    void weights(std::vector<double>* w); 
    void mlps(std::vector<double>* w); 

    std::vector<double> weight; 
    std::vector<double> masses; 
    std::vector<double> error; 

    std::vector<int> ntops; 
    std::vector<int> ntrus; 

    std::vector<std::vector<double>> mlp; 
    std::map<int, std::map<int, int>> matrix; 
}; 

struct pairs_t {
    pairs_t() = default;
    evn_t raw; 
    evn_t adj; 
}; 

struct performance_t {
    performance_t() = default; 

    std::map<std::string, pairs_t> truth; 
    std::map<std::string, pairs_t> nominal; 
    std::map<std::string, pairs_t> unmasked;
    std::map<std::string, pairs_t> masked; 
};

#endif
