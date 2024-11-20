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

    for (size_t x(0); x < pt.size(); ++x){
        g* prt     = new g(); 
        prt -> pt  = pt[x]; 
        prt -> eta = eta[x]; 
        prt -> phi = phi[x]; 
        prt -> e   = en[x]; 
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
        g* prt      = new g(); 
        prt -> pt   = pt[x]; 
        prt -> eta  = eta[x]; 
        prt -> phi  = phi[x]; 
        prt -> mass = en[x]; 
        out -> push_back(prt); 
    }
}; 


class zprime: public particle_template
{
    public:
        zprime(); 
        virtual ~zprime(); 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
}; 

class top: public particle_template
{
    public:
        top(); 
        virtual ~top(); 

        bool from_res = false; 
        bool is_hadronic = false; 

        std::vector<particle_template*> Children = {}; 
        std::vector<particle_template*> Jets = {}; 
        std::vector<particle_template*> Leptons = {}; 
        std::vector<particle_template*> Neutrinos = {}; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
};

class parton_w2: public particle_template
{
    public:
        parton_w2(); 
        virtual ~parton_w2(); 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
}; 


class parton_w1: public particle_template
{
    public:
        parton_w1(); 
        virtual ~parton_w1(); 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
}; 


class parton_b: public particle_template
{
    public:
        parton_b(); 
        virtual ~parton_b(); 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
}; 

class truthjet: public particle_template
{
    public:
        truthjet(); 
        virtual ~truthjet(); 

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
        bool btag_65 = false; 
        bool btag_70 = false; 
        bool btag_77 = false; 
        bool btag_85 = false; 
        bool btag_90 = false; 
        int flav = -1; 

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



#endif
