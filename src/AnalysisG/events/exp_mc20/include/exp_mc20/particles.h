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
        ~top() override; 

        int barcode; 
        int status; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 

}; 

class child: public particle_template
{
    public:
        child(); 
        ~child() override; 

        int barcode; 
        int status; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 

}; 

class physics_detector: public particle_template
{
    public:
        physics_detector(); 
        ~physics_detector() override; 

        std::vector<int> top_index = {}; 

        int parton_label = 0; 
        int cone_label = 0; 

        cproperty<bool, physics_detector> is_jet; 
        cproperty<bool, physics_detector> is_lepton; 
        cproperty<bool, physics_detector> is_photon; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
    
    private:
        std::vector<int> particle_type = {}; 
        static void get_type_jet(bool*, physics_detector*); 
        static void get_type_lepton(bool*, physics_detector*); 
        static void get_type_photon(bool*, physics_detector*); 
}; 

class physics_truth: public particle_template
{
    public:
        physics_truth(); 
        ~physics_truth() override; 

        std::vector<int> top_index = {}; 

        int parton_label = 0; 
        int cone_label = 0; 

        cproperty<bool, physics_truth> is_jet; 
        cproperty<bool, physics_truth> is_lepton; 
        cproperty<bool, physics_truth> is_photon; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
    
    private:
        std::vector<int> particle_type = {}; 
        static void get_type_jet(bool*, physics_truth*); 
        static void get_type_lepton(bool*, physics_truth*); 
        static void get_type_photon(bool*, physics_truth*); 
}; 

class electron: public particle_template
{
    public: 
        electron();
        ~electron() override;

        float d0 = 0; 
        int true_type = 0; 
        float delta_z0 = 0; 
        int true_origin = 0; 
        bool is_tight = false; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override;
}; 


class muon: public particle_template
{
    public: 
        muon();
        ~muon() override;

        float d0 = 0; 
        int true_type = 0; 
        float delta_z0 = 0; 
        int true_origin = 0; 
        bool is_tight = false; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override;
}; 

class jet: public particle_template
{
    public: 
        jet();
        ~jet() override;   

        bool btag_65; 
        bool btag_70; 
        bool btag_77; 
        bool btag_85; 
        bool btag_90; 

        int flav; 
        int label; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override;

}; 

#endif