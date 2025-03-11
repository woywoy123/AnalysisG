#ifndef NEUTRINO_COMBINATORIAL_H
#define NEUTRINO_COMBINATORIAL_H

#include <templates/selection_template.h>
#include <bsm_4tops/event.h>
#include <pyc/pyc.h>

class neutrino; 

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

    std::vector<neutrino*> truth_neutrinos; 

    // combinatorial
    std::vector<std::vector<neutrino*>> cobs_neutrinos; 
    std::vector<std::vector<neutrino*>> cmet_neutrinos; 
   
    // reference  
    std::vector<std::vector<neutrino*>> robs_neutrinos; 
    std::vector<std::vector<neutrino*>> rmet_neutrinos; 

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

        double masstop = 172.62*1000; 
        double massw   = 80.385*1000; 
        int    steps   = 100; 

    private:
        std::vector<std::vector<neutrino*>> build_nus(
                std::vector<particle_template*>* bqs, std::vector<particle_template*>* leps, 
                double met, double phi
        ); 

        std::vector<std::vector<neutrino*>> get_baseline(
                std::vector<particle_template*>* bqs, std::vector<particle_template*>* lpt, 
                std::vector<double>* tps, std::vector<double>* wbs, double met, double phi
        ); 
 

        template <typename g1, typename g2>
        std::vector<std::vector<neutrino*>> get_neutrinos(std::vector<g1>* bquark, std::vector<g2>* leps, double met, double phi){
            std::vector<particle_template*> bq; 
            this -> downcast(bquark, &bq); 
            std::vector<particle_template*> lp; 
            this -> downcast(leps, &lp); 
            return this -> build_nus(&bq, &lp, met, phi); 
        }
};

#endif
