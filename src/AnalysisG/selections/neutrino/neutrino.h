#ifndef neutrino_H
#define neutrino_H

#include <bsm_4tops/event.h>
#include <templates/selection_template.h>

class nu: public particle_template
{
    public:
        using particle_template::particle_template; 
        double nusol_tmass = 0; 
        double nusol_wmass = 0; 
        double exp_tmass = 0; 
        double exp_wmass = 0; 
        double min = 0; 
        double leppid = -99; 
        int idx = -1; 
}; 


class neutrino: public selection_template
{
    public:
        neutrino();
        virtual ~neutrino(); 
        selection_template* clone(); 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;

        std::map<std::string, double> delta_met; 
        std::map<std::string, double> delta_metnu; 
        std::map<std::string, double> obs_met; 
        std::map<std::string, double> nus_met; 

        std::map<std::string, double> dist_nu; 

        std::map<std::string, std::vector<int>> pdgid; 
        std::map<std::string, std::vector<double>> tru_topmass; 
        std::map<std::string, std::vector<double>> tru_wmass; 

        std::map<std::string, std::vector<double>> nusol_tmass; 
        std::map<std::string, std::vector<double>> nusol_wmass; 

        std::map<std::string, std::vector<double>> exp_topmass; 
        std::map<std::string, std::vector<double>> exp_wmass; 


        double scale   = 1000;
        double masstop = 172.62*scale; 
        double massw   = 80.385*scale; 
        int    steps   = 10; 

    private:

        std::vector<nu> build_nus(
                std::vector<long>* isb_, std::vector<long>* isl_,
                std::vector<double>* pt_, std::vector<double>* eta_, 
                std::vector<double>* phi_, std::vector<double>* energy_,
                double met, double phi, double scale); 


        template<typename g>
        std::vector<g*> upcast(std::vector<particle_template*>* inpt){
            typename std::vector<g*> out; 
            for (size_t x(0); x < inpt -> size(); ++x){
                out.push_back((g*)inpt -> at(x));  
            }
            return out; 
        }


        template<typename g>
        std::vector<g*> upcast(std::map<std::string, particle_template*>* inpt){
            std::vector<particle_template*> out = this -> vectorize(inpt); 
            return this -> upcast<g>(&out); 
        }

        template<typename g>
        void fill(
                std::vector<double>* pt , std::vector<double>* eta, 
                std::vector<double>* phi, std::vector<double>* energy, 
                std::vector<long>* isb   , std::vector<long>* isl, 
                std::vector<g*>* particles, double scale
        ){
            for (size_t x(0); x < particles -> size(); ++x){
                g* tr = particles -> at(x); 
                energy -> push_back(tr -> e / scale); 
                pt -> push_back(tr -> pt / scale); 
                eta -> push_back(tr -> eta); 
                phi -> push_back(tr -> phi); 
                isl -> push_back(long(tr -> is_lep)); 
                isb -> push_back(long(tr -> is_b)); 
            } 
        }

        template <typename g1, typename g2>
        std::vector<nu> get_neutrinos(
                std::vector<g1>* leps, std::vector<g2>* bquark, 
                double met, double phi, double scale
        ){
            std::vector<int> pidlep_;
            for (size_t x(0); x < leps -> size(); ++x){pidlep_.push_back((*leps)[x] -> pdgid);}

            std::vector<long> isb_, isl_; 
            std::vector<double> pt_, eta_, phi_, energy_; 
            this -> fill(&pt_, &eta_, &phi_, &energy_, &isb_, &isl_, leps, scale); 
            this -> fill(&pt_, &eta_, &phi_, &energy_, &isb_, &isl_, bquark, scale); 
            std::vector<nu> m = this -> build_nus(&isb_, &isl_, &pt_, &eta_, &phi_, &energy_, met, phi, scale); 
            for (size_t x(0); x < m.size(); ++x){m[x].leppid = pidlep_[m[x].idx];}
            return m; 
        }



};

#endif
