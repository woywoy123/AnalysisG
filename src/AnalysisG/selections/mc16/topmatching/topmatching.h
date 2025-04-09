#ifndef MC16_TOPMATCHING_H
#define MC16_TOPMATCHING_H

#include <bsm_4tops/event.h>
#include <templates/selection_template.h>

class topmatching: public selection_template
{
    public:
        topmatching();
        ~topmatching() override;
        selection_template* clone() override;

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override; 
        void merge(selection_template* sl) override; 

        std::vector<float> truth_top = {}; 
        std::vector<int> no_children = {}; 

        std::vector<float> topchildren_mass = {}; 
        std::vector<int>   topchildren_leptonic = {}; 

        std::vector<float> toptruthjets_mass     = {}; 
        std::vector<int>   toptruthjets_leptonic = {}; 
        std::vector<int>   toptruthjets_njets    = {}; 

        std::vector<float> topjets_children_mass     = {}; 
        std::vector<int>   topjets_children_leptonic = {}; 

        std::vector<float> topjets_leptons_mass     = {}; 
        std::vector<int>   topjets_leptons_leptonic = {}; 
        std::vector<int>   topjets_leptons_pdgid    = {}; 
        std::vector<int>   topjets_leptons_njets    = {}; 


        template <typename T>
        std::vector<T*> vectorize(std::map<std::string, T*>* mp){
            typename std::vector<T*> out = {}; 
            typename std::map<std::string, T*>::iterator itr = mp -> begin();
            for (; itr != mp -> end(); ++itr){out.push_back(itr -> second);}
            return out; 
        }

        template <typename T>
        std::vector<particle_template*> downcast(std::vector<T*>* mp){
            std::vector<particle_template*> out = {}; 
            for (size_t x(0); x < mp -> size(); ++x){
                out.push_back((particle_template*)mp -> at(x));
            }
            return out; 
        }
}; 

#endif
