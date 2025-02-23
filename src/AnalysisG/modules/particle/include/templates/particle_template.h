#ifndef PARTICLETEMPLATE_H
#define PARTICLETEMPLATE_H

#include <structs/particles.h>
#include <structs/property.h>
#include <structs/element.h>
#include <tools/tools.h>

#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cmath>

class particle_template : public tools
{
    public:
        particle_template();
        virtual ~particle_template(); 

        explicit particle_template(particle_t* p);
        explicit particle_template(double px, double py, double pz, double e); 
        explicit particle_template(double px, double py, double pz);

        void to_cartesian(); 
        void to_polar(); 

        cproperty<double, particle_template> e; 
        void static set_e(double*, particle_template*); 
        void static get_e(double*, particle_template*); 

        cproperty<double, particle_template> mass; 
        void static set_mass(double*, particle_template*); 
        void static get_mass(double*, particle_template*); 

        cproperty<double, particle_template> pt; 
        void static set_pt(double*, particle_template*); 
        void static get_pt(double*, particle_template*); 

        cproperty<double, particle_template> eta; 
        void static set_eta(double*, particle_template*); 
        void static get_eta(double*, particle_template*); 

        cproperty<double, particle_template> phi; 
        void static set_phi(double*, particle_template*); 
        void static get_phi(double*, particle_template*); 


        cproperty<double, particle_template> px; 
        void static set_px(double*, particle_template*); 
        void static get_px(double*, particle_template*); 

        cproperty<double, particle_template> py; 
        void static set_py(double*, particle_template*); 
        void static get_py(double*, particle_template*); 

        cproperty<double, particle_template> pz; 
        void static set_pz(double*, particle_template*); 
        void static get_pz(double*, particle_template*); 

        cproperty<int, particle_template> pdgid; 
        void static set_pdgid(int*, particle_template*); 
        void static get_pdgid(int*, particle_template*); 

        cproperty<std::string, particle_template> symbol; 
        void static set_symbol(std::string*, particle_template*); 
        void static get_symbol(std::string*, particle_template*); 

        cproperty<double, particle_template> charge; 
        void static set_charge(double*, particle_template*); 
        void static get_charge(double*, particle_template*); 

        cproperty<std::string, particle_template> hash; 
        void static get_hash(std::string*, particle_template*); 

        bool is(std::vector<int> p); 
        cproperty<bool, particle_template> is_b; 
        void static get_isb(bool*, particle_template*); 

        cproperty<bool, particle_template> is_lep; 
        void static get_islep(bool*, particle_template*); 

        cproperty<bool, particle_template> is_nu; 
        void static get_isnu(bool*, particle_template*); 

        cproperty<bool, particle_template> is_add; 
        void static get_isadd(bool*, particle_template*); 

        cproperty<bool, particle_template> lep_decay; 
        void static get_lepdecay(bool*, particle_template*); 

        cproperty<std::map<std::string, particle_template*>, particle_template> parents; 
        void static set_parents(std::map<std::string, particle_template*>*, particle_template*); 
        void static get_parents(std::map<std::string, particle_template*>*, particle_template*); 

        cproperty<std::map<std::string, particle_template*>, particle_template> children; 
        void static set_children(std::map<std::string, particle_template*>*, particle_template*); 
        void static get_children(std::map<std::string, particle_template*>*, particle_template*); 

        cproperty<std::string, particle_template> type; 
        void static set_type(std::string*, particle_template*); 
        void static get_type(std::string*, particle_template*); 

        cproperty<int, particle_template> index; 
        void static set_index(int*, particle_template*); 
        void static get_index(int*, particle_template*); 

        double DeltaR(particle_template* p);

        bool operator == (particle_template& p); 

        template <typename g>
        g operator + (g& p){
            p.to_cartesian(); 
            g p2 = g(); 
            p2.px = p.px + this -> px; 
            p2.py = p.py + this -> py;  
            p2.pz = p.pz + this -> pz; 
            p2.e  = p.e  + this -> e; 
            p2.to_cartesian(); 
            p2.to_polar(); 
            p2.data.type = this -> data.type; 
            return p2; 
        }

        void operator += (particle_template* p); 
        void iadd(particle_template* p); 
      
        bool register_parent(particle_template* p);
        std::map<std::string, particle_template*> m_parents; 

        bool register_child(particle_template* p);
        std::map<std::string, particle_template*> m_children; 

        void add_leaf(std::string key, std::string leaf = ""); 
        std::map<std::string, std::string> leaves = {}; 

        void apply_type_prefix(); 

        virtual void build(std::map<std::string, particle_template*>* event, element_t* el); 
        virtual particle_template* clone(); 

        particle_t data;  
}; 
#endif

