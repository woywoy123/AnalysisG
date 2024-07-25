#ifndef EVENTS_SSML_MC20_H
#define EVENTS_SSML_MC20_H

#include <ssml_mc20/particles.h>
#include <templates/event_template.h>

class ssml_mc20: public event_template
{
    public:
        ssml_mc20(); 
        ~ssml_mc20() override; 

        std::vector<particle_template*> Jets = {}; 
        std::vector<particle_template*> Leptons = {}; 
        std::vector<particle_template*> Electrons = {}; 
        std::vector<particle_template*> Muons = {}; 

        float met;
        float phi; 
        float met_sum; 
        unsigned long long eventNumber; 

        int n_bj_65;
        int n_bj_70; 
        int n_bj_77; 
        int n_bj_85; 
        int n_bj_90; 

        int n_els; 
        int n_leps; 
        int n_mus; 
        int n_jets; 

        event_template* clone() override; 
        void build(element_t* el) override; 
        void CompileEvent() override; 

    private: 
        std::map<std::string, jet*> m_jets = {}; 
        std::map<std::string, muon*> m_muons = {}; 
        std::map<std::string, lepton*> m_leptons = {}; 
        std::map<std::string, electron*> m_electrons = {}; 

        template <typename g>
        void vectorize(std::map<std::string, g*>* inp, std::vector<particle_template*>* out){
            typename std::map<std::string, g*>::iterator itr = inp -> begin(); 
            for (; itr != inp -> end(); ++itr){
                out -> push_back((particle_template*)itr -> second);
            }
        }


}; 


#endif
