#ifndef EVENTS_EXP_MC20_H
#define EVENTS_EXP_MC20_H

#include <exp_mc20/particles.h>
#include <templates/event_template.h>

class exp_mc20: public event_template
{
    public:
        exp_mc20(); 
        ~exp_mc20() override; 

        std::vector<particle_template*> Tops; 
        std::vector<particle_template*> TruthChildren; 
        std::vector<particle_template*> PhysicsTruth;
        std::vector<particle_template*> Jets; 
        std::vector<particle_template*> Leptons; 
        std::vector<particle_template*> PhysicsDetector; 
        std::vector<particle_template*> Detector; 

        unsigned long long event_number = 0; 
        float met_sum = 0; 
        float met = 0; 
        float phi = 0; 
        float mu = 0; 

        event_template* clone() override; 
        void build(element_t* el) override; 
        void CompileEvent() override;  
    
    private:
        std::map<std::string, top*> m_tops; 
        std::map<std::string, child*> m_children; 
        std::map<std::string, physics_detector*> m_physdet; 
        std::map<std::string, physics_truth*> m_phystru; 

        std::map<std::string, electron*> m_electrons; 
        std::map<std::string, muon*> m_muons; 
        std::map<std::string, jet*> m_jets; 

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
