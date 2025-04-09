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
        float top_mass_contribution(std::vector<g*> jts, std::map<std::string, std::vector<float>>* fracx, gx* p, top* ti, float cut, bool cut_gluon){
            std::string top_i = ti -> hash; 

            typename std::vector<g*> passed; 
            for (size_t tn(0); tn < jts.size(); ++tn){
                typename std::vector<gx*> tjp = jts[tn] -> Parton; 
                if (tjp.size() == 0){continue;}

                bool has_gluon = false; 
                typename std::map<std::string, std::vector<gx*>> parton_to_top; 
                for (size_t tx(0); tx < tjp.size(); ++tx){
                    std::vector<top*> tti; 
                    std::vector<top_children*> tch; 

                    top* cur_t = nullptr; 
                    this -> upcast(&tjp[tx] -> parents, &tch); 
                    for (size_t ci(0); ci < tch.size(); ++ci){
                        this -> upcast(&tch[ci] -> parents, &tti);
                        tti = this -> make_unique(&tti); 
                        for (size_t x(0); x < tti.size(); ++x){
                            if (top_i != std::string(tti[x] -> hash)){continue;}
                            cur_t = tti[x]; break; 
                        }
                        if (!cur_t){continue;}
                        break;
                    }
                    
                    if (!cur_t){continue;}
                    parton_to_top[top_i].push_back(tjp[tx]); 
                    has_gluon += std::abs(tjp[tx] -> pdgid) == 21; 
                }
                if (!parton_to_top[top_i].size()){continue;}
                parton_to_top[top_i] = this -> make_unique(&parton_to_top[top_i]);

                particle_template* tsm = nullptr; 
                this -> sum(&parton_to_top[top_i], &tsm); 
                float r = tsm -> e / jts[tn] -> e;

                if (fracx){(*fracx)[this -> to_string(jts[tn] -> Tops.size()) + "::tops"].push_back(tsm -> e / jts[tn] -> e);}
                if (r >= cut && !cut_gluon){passed.push_back(jts[tn]); continue;}
                if (cut == 0 && !has_gluon){passed.push_back(jts[tn]); continue;}
                if (r >= cut && cut_gluon && has_gluon){passed.push_back(jts[tn]); continue;}
                if (has_gluon){continue;}
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
