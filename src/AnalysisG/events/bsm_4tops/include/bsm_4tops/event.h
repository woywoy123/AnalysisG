#ifndef EVENTS_BSM4TOPS_H
#define EVENTS_BSM4TOPS_H

#include <bsm_4tops/particles.h>
#include <templates/event_template.h>

class bsm_4tops: public event_template
{
    public:
        bsm_4tops(); 
        ~bsm_4tops() override; 

        std::vector<particle_template*> Tops = {}; 
        std::vector<particle_template*> Children = {}; 
        std::vector<particle_template*> TruthJets = {}; 

        std::vector<particle_template*> Jets = {}; 
        std::vector<particle_template*> Electrons = {}; 
        std::vector<particle_template*> Muons = {};  
        std::vector<particle_template*> DetectorObjects = {};

        unsigned long long event_number = 0; 
        float mu = 0; 
        float met = 0; 
        float phi = 0; 

        event_template* clone() override; 
        void build(element_t* el) override; 
        void CompileEvent() override; 

    private: 
        std::map<std::string, top*>            m_Tops = {}; 
        std::map<std::string, top_children*>   m_Children = {}; 
        std::map<std::string, truthjet*>       m_TruthJets = {}; 
        std::map<std::string, jet*>            m_Jets = {}; 
        std::map<std::string, electron*>       m_Electrons = {}; 
        std::map<std::string, muon*>           m_Muons = {};  
        std::map<std::string, jetparton*>      m_JetParton = {}; 
        std::map<std::string, truthjetparton*> m_TruthJetParton = {};  

        template <typename G>
        std::map<int, G*> sort_by_index(std::map<std::string, G*>* ipt){
            std::map<int, G*> data = {}; 
            typename std::map<std::string, G*>::iterator ix; 
            ix = ipt -> begin();
            for (; ix != ipt -> end(); ++ix){
                data[(int)ix -> second -> index] = ix -> second;
            }
            return data; 
        }

        template <typename m, typename G>
        void vectorize(std::map<m, G*>* ipt, std::vector<particle_template*>* vec){
            typename std::map<m, G*>::iterator ix = ipt -> begin();
            for (; ix != ipt -> end(); ++ix){vec -> push_back(ix -> second);}
        }



}; 


#endif
