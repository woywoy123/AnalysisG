#ifndef PARTICLES_SSML_MC20_H
#define PARTICLES_SSML_MC20_H

#include <templates/particle_template.h>

class jet: public particle_template
{
    public:
        jet(); 
        ~jet() override; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 

        bool gn2_btag_65;
        bool gn2_btag_70;
        bool gn2_btag_77;
        bool gn2_btag_85;
        bool gn2_btag_90;
}; 

class lepton: public particle_template
{
    public:
        lepton(); 
        ~lepton() override; 

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

class electron: public particle_template
{
    public:
        electron(); 
        ~electron() override; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 

}; 



#endif
