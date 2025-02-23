#ifndef PARTICLES_SSML_MC20_H
#define PARTICLES_SSML_MC20_H

#include <templates/particle_template.h>

template <typename g>
void pmu(std::vector<g*>* out, element_t* el){
    std::vector<float> pt, eta, phi, en;  
    el -> get("pt"    , &pt); 
    el -> get("eta"   , &eta); 
    el -> get("phi"   , &phi); 
    el -> get("energy", &en); 

    for (int x(0); x < pt.size(); ++x){
        g* prt       = new g(); 
        prt -> pt    = pt[x]; 
        prt -> eta   = eta[x]; 
        prt -> phi   = phi[x]; 
        prt -> e     = en[x]; 
        prt -> index = x; 
        out -> push_back(prt); 
    }
}; 

template <typename g>
void pmu_mass(std::vector<g*>* out, element_t* el){
    std::vector<float> pt, eta, phi, en;  
    el -> get("pt"  , &pt); 
    el -> get("eta" , &eta); 
    el -> get("phi" , &phi); 
    el -> get("mass", &en); 

    for (size_t x(0); x < pt.size(); ++x){
        if (en[x] < 0){break;}
        g* prt       = new g(); 
        prt -> pt    = pt[x]; 
        prt -> eta   = eta[x]; 
        prt -> phi   = phi[x]; 
        prt -> mass  = en[x]; 
        prt -> index = x; 
        out-> push_back(prt); 
    }
}; 


class zboson: public particle_template
{
    public:
        zboson(); 
        virtual ~zboson(); 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
}; 

class top: public particle_template
{
    public:
        top(); 
        virtual ~top(); 

        bool from_res = false; 

        std::vector<particle_template*> jets = {}; 
        std::vector<particle_template*> leptons = {}; 
        std::vector<particle_template*> truthjets = {}; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
};

class truthjet: public particle_template
{
    public:
        truthjet(); 
        virtual ~truthjet(); 

        int top_index = -1; 
        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
}; 

class jet: public particle_template
{
    public:
        jet(); 
        virtual ~jet(); 

        cproperty<bool, jet> from_res; 

        int top_index = -1; 
        bool btag_77 = false; 
        bool btag_85 = false; 
        bool btag_90 = false; 
        bool sel_85  = false; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 

    public:
        static void get_from_res(bool* val, jet* el);
}; 

class electron: public particle_template
{
    public:
        electron(); 
        virtual ~electron(); 

        cproperty<bool, electron> from_res; 
        int top_index = -1; 
        int pass_ecids = -1; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 

    public:
        static void get_from_res(bool* val, electron* el);

}; 

class muon: public particle_template
{
    public:
        muon(); 
        virtual ~muon(); 

        int top_index = -1; 
        cproperty<bool, muon> from_res; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 

    public:
        static void get_from_res(bool* val, muon* el);
}; 

class parton: public particle_template
{
    public:
        parton(); 
        virtual ~parton(); 
        int top_index      = -1; 
        int jet_index      = -1; 
        int truthjet_index = -1; 

        int muon_index     = -1; 
        int electron_index = -1; 

        std::map<std::string, jet*> jets; 
        std::map<std::string, truthjet*> truthjets; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
}; 

class lepton: public particle_template
{
    public:
        lepton(); 
        virtual ~lepton();

        int ambiguity = 99; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override;  
}; 





#endif
