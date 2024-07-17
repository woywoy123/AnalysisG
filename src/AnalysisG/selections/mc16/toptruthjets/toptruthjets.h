#ifndef TOPTRUTHJETS_H
#define TOPTRUTHJETS_H

#include <bsm_4tops/event.h>
#include <templates/selection_template.h>

class toptruthjets: public selection_template
{
    public:
        toptruthjets();
        ~toptruthjets() override; 
        selection_template* clone() override; 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;

    private:
        std::map<std::string, std::map<std::string, std::map<std::string, std::vector<float>>>> top_mass = {}; 
        std::map<std::string, std::map<std::string, std::vector<float>>> truthjet_top = {}; 
        std::vector<int> ntops_lost = {}; 


        std::vector<particle_template*> make_unique(std::vector<particle_template*>* inpt);


        template <typename g>
        void downcast(std::vector<g*>* inpt, std::vector<particle_template*>* out){
            for (size_t x(0); x < inpt -> size(); ++x){
                out -> push_back((particle_template*)(*inpt)[x]);
            }
        }

        template <typename g>
        void get_leptonics(std::map<std::string, g*> inpt, std::vector<particle_template*>* out){
            typename std::map<std::string, g*>::iterator itr = inpt.begin(); 
            for (; itr != inpt.end(); ++itr){
                if (!itr -> second -> is_lep && !itr -> second -> is_nu){continue;}
                out -> push_back((particle_template*)itr -> second);
            }
        }

};

#endif
