#ifndef MET_MC16_H
#define MET_MC16_H

#include <templates/selection_template.h>

class neutrino; 

struct angle_t {
    std::vector<double> pln_w1 = {}; 
    std::vector<double> pln_t1 = {}; 

    std::vector<double> pln_w2 = {}; 
    std::vector<double> pln_t2 = {}; 
    std::vector<double> pln_t1xt2 = {};

    double dR_nu_lep1 = 0;
    double dR_nu_lep2 = 0;

    bool has_null = true; 
};

struct packet_t {
    double missing_evn_px = 0; 
    double missing_evn_py = 0; 

    double missing_det_px = 0; 
    double missing_det_py = 0; 
    double missing_det_pz = 0; 

    double missing_nus_px = 0; 
    double missing_nus_py = 0; 
    double missing_nus_pz = 0; 

    double num_neutrino     = 0;
    double num_leptons      = 0;  
    double num_leptons_reco = 0; 

    std::vector<int>       top_index  = {}; 
    std::vector<double> mass_tru_top  = {}; 
    std::vector<double> mass_tru_top3 = {}; 
    std::vector<particle_template*> tru_nu  = {}; 

    std::vector<double> chi2_sols = {}; 
    std::vector<int> top_index_sols1 = {}; 
    std::vector<int> top_index_sols2 = {}; 

    std::vector<double> top_mass_sols1 = {}; 
    std::vector<double> top_mass_sols2 = {}; 

    std::vector<neutrino*> nu1 = {}; 
    std::vector<neutrino*> nu2 = {}; 
    std::vector<angle_t>  agnR = {}; 
    std::vector<angle_t>  agnT = {}; 
};

class met: public selection_template
{
    public:
        met();
        ~met() override; 
        selection_template* clone() override; 

        std::map<int, std::map<std::string, particle_template*>> match_tops(std::vector<particle_template*> tops, std::vector<particle_template*> leps); 
        std::tuple<std::vector<neutrino*>, std::vector<neutrino*>> reconstruction(std::vector<particle_template*> nodes, double met_, double phi_); 

        std::vector<double> sum_cart(std::vector<particle_template*> ptx); 
        std::vector<double> cross(std::vector<double> v1, std::vector<double> v2, bool norm); 
        std::vector<double> pmc(particle_template* p1); 

        double chi2(particle_template* nut, particle_template* nur); 
        double cart_px(std::vector<particle_template*> prt);
        double cart_py(std::vector<particle_template*> prt);
        double cart_pz(std::vector<particle_template*> prt);
        double cart_px(double met, double phi);
        double cart_py(double met, double phi);

        angle_t angle(
                particle_template* nu1, particle_template* l1, particle_template* b1, 
                particle_template* nu2, particle_template* l2, particle_template* b2, 
                std::vector<double> det_, std::vector<double> met_
        ); 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;

        int    steps    = 20; 
        double perturb  = 1e-2; 
        double distance = 1e-3; 
        double masstop  = 172.62*1000; 
        double massw    = 80.385*1000; 
    
    private:
        template <typename g>
        void add(std::map<std::string, particle_template*>* out, std::vector<g*>* in){
            for (size_t x(0); x < in -> size(); ++x){(*out)[in -> at(x) -> hash] = (particle_template*)in -> at(x);}
        }

        std::vector<particle_template*> storage = {}; 
        packet_t data; 
};

#endif
