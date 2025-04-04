#ifndef EVENTS_PARTICLES_EXP_MC20_H
#define EVENTS_PARTICLES_EXP_MC20_H

#include <templates/particle_template.h>

template <typename g>
void pmu(std::vector<g*>* out, element_t* el){
    std::vector<float> pt, eta, phi, en, ch;  
    el -> get("pt"    , &pt); 
    el -> get("eta"   , &eta); 
    el -> get("phi"   , &phi); 
    el -> get("energy", &en); 

    for (size_t x(0); x < pt.size(); ++x){
        g* prt     = new g(); 
        prt -> pt  = pt[x]; 
        prt -> eta = eta[x]; 
        prt -> phi = phi[x]; 
        prt -> e   = en[x]; 
        out -> push_back(prt); 
    }
}

class top: public particle_template
{
    public:
        top(); 
        virtual ~top(); 

        int top_index = -1; 
        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 

}; 

class child: public particle_template
{
    public:
        child(); 
        virtual ~child(); 

        std::vector<int> top_index = {}; 
        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 

}; 

class physics_detector: public particle_template
{
    public:
        physics_detector(); 
        virtual ~physics_detector(); 

        std::vector<int> top_index = {}; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
}; 

class physics_truth: public particle_template
{
    public:
        physics_truth(); 
        virtual ~physics_truth(); 

        std::vector<int> top_index = {}; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
}; 

class electron: public particle_template
{
    public: 
        electron();
        virtual ~electron();

        float d0 = 0; 
        int true_type = 0; 
        float delta_z0 = 0; 
        int true_origin = 0; 
        bool is_tight = false; 
        std::vector<int> top_index = {}; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override;
}; 


class muon: public particle_template
{
    public: 
        muon();
        virtual ~muon();

        float d0 = 0; 
        int true_type = 0; 
        float delta_z0 = 0; 
        int true_origin = 0; 
        bool is_tight = false; 
        std::vector<int> top_index = {}; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override;
}; 

class jet: public particle_template
{
    public: 
        jet();
        virtual ~jet();   

        bool btag_65; 
        bool btag_70; 
        bool btag_77; 
        bool btag_85; 
        bool btag_90; 

        int flav; 
        int label; 

        std::vector<int> top_index = {}; 
        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override;

}; 

#endif
