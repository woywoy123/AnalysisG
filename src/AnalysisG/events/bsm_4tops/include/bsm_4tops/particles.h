#ifndef EVENTS_BSM4TOPS_PARTICLES_H
#define EVENTS_BSM4TOPS_PARTICLES_H

#include <templates/particle_template.h>

class truthjetparton; 
class jetparton; 
class truthjet; 
class jet; 

template <typename g>
void assign_vector(std::vector<g*>* inpt, element_t* el){
    std::vector<int> _index, _pdgid; 
    std::vector<float> _pt, _eta, _phi, _e; 
    
    el -> get("index"   , &_index); 
    el -> get("pdgid"   , &_pdgid);
    
    el -> get("pt" , &_pt); 
    el -> get("eta", &_eta);
    el -> get("phi", &_phi); 
    el -> get("e"  , &_e); 

    for (int x(0); x < _pt.size(); ++x){
        g* p          = new g();
        p -> pt       = _pt[x]; 
        p -> eta      = _eta[x]; 
        p -> phi      = _phi[x]; 
        p -> e        = _e[x]; 
        p -> index    = _index[x]; 
        p -> pdgid    = _pdgid[x]; 
        inpt -> push_back(p); 
    }
};



class top: public particle_template
{
    public:
       
        top(); 
        ~top() override; 

        bool from_res = false; 
        int status = -1; 

        std::vector<truthjet*> TruthJets = {}; 
        std::vector<jet*> Jets = {}; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override;
}; 

class top_children: public particle_template
{
    public: 
        top_children();
        ~top_children() override;   

        int top_index = -1;  

        cproperty<bool, top_children> from_res; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override;

    private:
        void static get_from_res(bool*, top_children*);
}; 


class truthjet: public particle_template
{
    public: 
        truthjet();
        ~truthjet() override;   

        int top_quark_count = -1;
        int w_boson_count = -1; 
        std::vector<int> top_index = {}; 
        cproperty<bool, truthjet> from_res; 

        std::vector<top*> Tops = {};
        std::vector<truthjetparton*> Parton = {}; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override;

    private:
        void static get_from_res(bool*, truthjet*);

}; 

class truthjetparton: public particle_template
{
    public: 
        truthjetparton();
        ~truthjetparton() override;

        int truthjet_index;
        std::vector<int> topchild_index = {};

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override;
}; 


class jet: public particle_template
{
    public: 
        jet();
        ~jet() override;   

        std::vector<top*> Tops = {}; 
        std::vector<jetparton*> Parton = {}; 

        std::vector<int> top_index = {}; 
        bool btag_DL1r_60; 
        bool btag_DL1_60;  
        bool btag_DL1r_70; 
        bool btag_DL1_70;  
        bool btag_DL1r_77; 
        bool btag_DL1_77;  
        bool btag_DL1r_85; 
        bool btag_DL1_85;  

        float DL1_b; 
        float DL1_c;  
        float DL1_u; 
        float DL1r_b; 
        float DL1r_c; 
        float DL1r_u;

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override;
}; 


class jetparton: public particle_template
{
    public: 
        jetparton();
        ~jetparton() override;

        int jet_index;
        std::vector<int> topchild_index = {}; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override;
}; 


class electron: public particle_template
{
    public: 
        electron();
        ~electron() override;

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override;
}; 


class muon: public particle_template
{
    public: 
        muon();
        ~muon() override;

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override;
}; 

#endif
