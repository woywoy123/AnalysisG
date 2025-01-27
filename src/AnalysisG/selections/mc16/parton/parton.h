#ifndef PARTON_H
#define PARTON_H

#include <bsm_4tops/event.h>
#include <templates/selection_template.h>




class parton: public selection_template
{
    public:
        parton();
        ~parton() override; 
        selection_template* clone() override; 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;

        template <typename g, typename gx>
        float top_mass_contribution(std::vector<g*> jts, std::map<std::string, std::vector<float>>* fracx, gx* p, top* ti, float cut){
            typename std::vector<g*> passed; 
            for (size_t tn(0); tn < jts.size(); ++tn){
                typename std::vector<gx*> tjp = jts[tn] -> Parton; 
                if (tjp.size() == 0){continue;}

                particle_template* sm = nullptr; 
                this -> sum(&tjp, &sm); 

                typename std::map<std::string, std::vector<gx*>> parton_to_top; 
                for (size_t tx(0); tx < tjp.size(); ++tx){
                    std::vector<top*> tti; 
                    std::vector<top_children*> tch; 

                    this -> upcast(&tjp[tx] -> parents, &tch); 
                    for (size_t ci(0); ci < tch.size(); ++ci){this -> upcast(&tch[ci] -> parents, &tti);}
                    tti = this -> make_unique(&tti); 

                    if (tti.size() > 1 || tti.size() == 0){continue;}
                    std::string ket = tti[0] -> hash; 
                    parton_to_top[ket].push_back(tjp[tx]); 
                    parton_to_top[ket] = this -> make_unique(&parton_to_top[ket]); 
                }
                if (!parton_to_top[ti -> hash].size()){continue;}
                particle_template* tsm = nullptr; 
                this -> sum(&parton_to_top[ti -> hash], &tsm); 
                float r = tsm -> e / jts[tn] -> e;
                if (fracx){(*fracx)[this -> to_string(jts[tn] -> Tops.size()) + "::tops"].push_back(r);}
                if (r < cut){continue;}
                passed.push_back(jts[tn]); 
            } 
            return this -> sum(&passed); 
        }

        std::map<std::string, std::vector<float>> ntops_tjets_pt; 
        std::map<std::string, std::vector<float>> ntops_tjets_e; 

        std::map<std::string, std::vector<float>> ntops_jets_pt; 
        std::map<std::string, std::vector<float>> ntops_jets_e; 

        std::map<std::string, std::vector<float>> nparton_tjet_e; 
        std::map<std::string, std::vector<float>> nparton_jet_e; 
 
        std::map<std::string, std::vector<float>> frac_parton_tjet_e; 
        std::map<std::string, std::vector<float>> frac_parton_jet_e;

        std::map<std::string, std::vector<float>> frac_ntop_tjet_contribution;  
        std::map<std::string, std::vector<float>> frac_ntop_jet_contribution;  
        std::map<std::string, std::vector<float>> frac_mass_top; 
};

#endif
