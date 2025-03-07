#ifndef NEUTRINO_COMBINATORIAL_H
#define NEUTRINO_COMBINATORIAL_H

#include <bsm_4tops/event.h>
#include <templates/selection_template.h>

class nu: public particle_template
{
    public:
        using particle_template::particle_template; 
        double exp_tmass = 0; 
        double exp_wmass = 0; 
        double min = 0; 
        long idx = -1; 
}; 


class particle: public particle_template
{
    public:
        using particle_template::particle_template;
}; 

struct event_data {
    double delta_met = 0; 
    double delta_metnu = 0; 
    double observed_met = 0; 
    double neutrino_met = 0; 
    std::vector<nu*> truth_neutrinos; 

    // combinatorial
    std::vector<nu*> cobs_neutrinos; 
    std::vector<nu*> cmet_neutrinos; 
   
    // reference  
    std::vector<nu*> robs_neutrinos; 
    std::vector<nu*> rmet_neutrinos; 

    std::vector<particle*> bquark; 
    std::vector<particle*> lepton; 
    std::vector<particle*> tops; 
}; 


class combinatorial: public selection_template
{
    public:
        combinatorial();
        virtual ~combinatorial(); 
        selection_template* clone(); 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;
        std::map<std::string, event_data> output; 

        double scale   = 1.0; //1000;
        double masstop = 172.62*1000; 
        double massw   = 80.385*1000; 
        int    steps   = 1000; 

    private:
        void fill(
                std::vector<double>* pt, std::vector<double>* eta, 
                std::vector<double>* phi, std::vector<double>* energy, 
                std::vector<long>*  isb, std::vector<long>* isl, 
                std::vector<particle_template*>* particles, double scale
        ); 


        std::vector<nu*> build_nus(
                std::vector<long>* isb_, std::vector<long>* isl_,
                std::vector<particle_template*>* bqs, std::vector<particle_template*>* leps, 
                double met, double phi
        ); 

        std::vector<nu*> get_baseline(
                std::vector<particle_template*>* bqs, std::vector<particle_template*>* lpt, 
                std::vector<double>* tps, std::vector<double>* wbs, double met, double phi
        ); 
 

        template <typename g>
        void fill(std::vector<long>* isb, std::vector<long>* isl, std::vector<g*>* particles){
            for (size_t x(0); x < particles -> size(); ++x){
                g* tr = (*particles)[x]; 
                isl -> push_back(long(tr -> is_lep)); 
                isb -> push_back(long(tr -> is_b)); 
            } 
        }

        template <typename g1, typename g2>
        std::vector<nu*> get_neutrinos(std::vector<g1>* bquark, std::vector<g2>* leps, double met, double phi){
            std::vector<int> pidlep_;
            for (size_t x(0); x < leps -> size(); ++x){pidlep_.push_back((*leps)[x] -> pdgid);}
            std::vector<long> isb_, isl_; 
            this -> fill(&isb_, &isl_, leps); 
            this -> fill(&isb_, &isl_, bquark); 
            std::vector<nu*> m = this -> build_nus(&isb_, &isl_, bquark, leps, met, phi); 
            for (size_t x(0); x < m.size(); ++x){m[x] -> pdgid = pidlep_[m[x] -> idx];}
            return m; 
        }
};

#endif
