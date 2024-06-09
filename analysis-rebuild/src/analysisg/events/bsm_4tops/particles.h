#ifndef EVENTS_BSM4TOPS_PARTICLES_H
#define EVENTS_BSM4TOPS_PARTICLES_H

#include <templates/particle_template.h>

//class truthjet_parton; 
class parton; 
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
        g* p = new g();
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
        ~top(); 

        bool from_res = false; 
        int status = -1; 

        std::vector<truthjet*> TruthJets = {}; 
        std::vector<jet*> Jets = {}; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override;
}; 

class children: public particle_template
{
    public: 
        children();
        ~children();   

        int top_index = -1;  
        cproperty<bool, children> from_res; 
        void static get_from_res(bool*, children*);

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override;

}; 


class truthjet: public particle_template
{
    public: 
        truthjet();
        ~truthjet();   

        int top_quark_count = -1;
        int w_boson_count = -1; 
        std::vector<int> top_index = {}; 

        std::vector<top*> Tops = {};
        //std::vector<truthjet_parton*> = {}; 

        cproperty<bool, truthjet> from_res; 
        void static get_from_res(bool*, truthjet*);

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override;

}; 






class jet: public particle_template
{
    public: 
        jet();
        ~jet();   
}; 


class parton: public particle_template
{
    public: 
        parton();
        ~parton();
}; 



#endif
