#ifndef EVENTS_SSML_MC20_H
#define EVENTS_SSML_MC20_H

#include <ssml_mc20/particles.h>
#include <templates/event_template.h>

class ssml_mc20: public event_template
{
    public:
        ssml_mc20(); 
        virtual ~ssml_mc20(); 

        unsigned long long eventNumber; 

        float met;
        float phi; 
        float met_sum; 

        int n_bj_65;
        int n_bj_70; 
        int n_bj_77; 
        int n_bj_85; 
        int n_bj_90; 

        int n_els; 
        int n_leps; 
        int n_mus; 
        int n_jets; 

        std::vector<particle_template*> Tops; 
        std::vector<particle_template*> TruthChildren; 

        std::vector<particle_template*> Jets; 
        std::vector<particle_template*> Leptons; 
        std::vector<particle_template*> Detector; 

        event_template* clone() override; 
        void build(element_t* el) override; 
        void CompileEvent() override; 

    private:
        std::map<std::string, zprime*>   m_zprime; 
        std::map<std::string, top*>      m_tops; 

        std::map<std::string, parton_w1*> m_parton1; 
        std::map<std::string, parton_w2*> m_parton2; 
        std::map<std::string, parton_b*>  m_bpartons; 
        std::map<std::string, truthjet*>  m_truthjets; 

        std::map<std::string, jet*>      m_jets; 
        std::map<std::string, muon*>     m_muons; 
        std::map<std::string, electron*> m_electrons; 

        template <typename G>
        std::map<int, G*> sort_by_index(std::map<std::string, G*>* ipt){
            std::map<int, G*> data = {}; 
            typename std::map<std::string, G*>::iterator ix = ipt -> begin();
            for (; ix != ipt -> end(); ++ix){data[int(ix -> second -> index)] = ix -> second;}
            return data; 
        }

        template <typename m, typename G, typename g>
        void vectorize(std::map<m, G*>* ipt, std::vector<g*>* vec){
            typename std::map<m, G*>::iterator ix = ipt -> begin();
            for (; ix != ipt -> end(); ++ix){vec -> push_back(ix -> second);}
        }



}; 


#endif
