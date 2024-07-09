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

        float sum(std::vector<particle_template*>* ch); 
        
        std::vector<float> truth_top = {}; 
        std::vector<int> no_children = {}; 
        std::map<std::string, std::vector<float>> truth_children = {}; 
        std::map<std::string, std::vector<float>> truth_jets = {}; 

        std::map<std::string, std::vector<float>> n_truth_jets_lep = {}; 
        std::map<std::string, std::vector<float>> n_truth_jets_had = {}; 

        std::map<std::string, std::vector<float>> jets_truth_leps = {}; 
        std::map<std::string, std::vector<float>> jet_leps = {}; 

        std::map<std::string, std::vector<float>> n_jets_lep = {}; 
        std::map<std::string, std::vector<float>> n_jets_had = {}; 


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
