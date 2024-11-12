#include <templates/particle_template.h>

particle_template::particle_template(){
    this -> e.set_setter(this -> set_e); 
    this -> e.set_getter(this -> get_e); 
    this -> e.set_object(this); 

    this -> mass.set_setter(this -> set_mass); 
    this -> mass.set_getter(this -> get_mass); 
    this -> mass.set_object(this); 

    this -> pt.set_setter(this -> set_pt); 
    this -> pt.set_getter(this -> get_pt); 
    this -> pt.set_object(this); 

    this -> eta.set_setter(this -> set_eta); 
    this -> eta.set_getter(this -> get_eta); 
    this -> eta.set_object(this); 

    this -> phi.set_setter(this -> set_phi); 
    this -> phi.set_getter(this -> get_phi); 
    this -> phi.set_object(this); 

    this -> px.set_setter(this -> set_px); 
    this -> px.set_getter(this -> get_px); 
    this -> px.set_object(this); 

    this -> py.set_setter(this -> set_py); 
    this -> py.set_getter(this -> get_py); 
    this -> py.set_object(this); 

    this -> pz.set_setter(this -> set_pz); 
    this -> pz.set_getter(this -> get_pz); 
    this -> pz.set_object(this); 

    this -> pdgid.set_setter(this -> set_pdgid); 
    this -> pdgid.set_getter(this -> get_pdgid); 
    this -> pdgid.set_object(this); 

    this -> symbol.set_setter(this -> set_symbol); 
    this -> symbol.set_getter(this -> get_symbol); 
    this -> symbol.set_object(this); 

    this -> charge.set_setter(this -> set_charge); 
    this -> charge.set_getter(this -> get_charge); 
    this -> charge.set_object(this); 

    this -> hash.set_getter(this -> get_hash); 
    this -> hash.set_object(this); 

    this -> is_b.set_getter(this -> get_isb); 
    this -> is_b.set_object(this); 

    this -> is_lep.set_getter(this -> get_islep); 
    this -> is_lep.set_object(this); 

    this -> is_nu.set_getter(this -> get_isnu); 
    this -> is_nu.set_object(this); 

    this -> is_add.set_getter(this -> get_isadd); 
    this -> is_add.set_object(this); 

    this -> lep_decay.set_getter(this -> get_lepdecay); 
    this -> lep_decay.set_object(this); 

    this -> parents.set_setter(this -> set_parents); 
    this -> parents.set_getter(this -> get_parents); 
    this -> parents.set_object(this); 

    this -> children.set_setter(this -> set_children); 
    this -> children.set_getter(this -> get_children); 
    this -> children.set_object(this); 

    this -> type.set_setter(this -> set_type); 
    this -> type.set_getter(this -> get_type); 
    this -> type.set_object(this); 

    this -> index.set_setter(this -> set_index); 
    this -> index.set_getter(this -> get_index); 
    this -> index.set_object(this); 
}

particle_template::particle_template(particle_t* p) : particle_template(){this -> data = *p;}

particle_template::particle_template(double px, double py, double pz, double e) : particle_template(){
    particle_t* p = &this -> data; 
    p -> px = px; p -> py = py; p -> pz = pz; p -> e = e; 
    p -> polar = true; 
}

particle_template::particle_template(double px, double py, double pz) : particle_template() {
    particle_t* p = &this -> data; 
    p -> px = px; p -> py = py; p -> pz = pz; this -> e; 
    p -> polar = true; 
}

particle_template::~particle_template(){}

void particle_template::operator += (particle_template* p){
    p -> to_cartesian(); 
    this -> to_cartesian();
    this -> data.px += p -> px; 
    this -> data.py += p -> py; 
    this -> data.pz += p -> pz; 
    this -> data.e  += p -> e; 
    this -> data.polar = true;
}

void particle_template::iadd(particle_template* p){
    *this += p; 
}

bool particle_template::operator == (particle_template& p){
    return this -> hash == p.hash; 
}

void particle_template::apply_type_prefix(){
    std::string _type = this -> type; 
    std::map<std::string, std::string> lf = {}; 
    std::map<std::string, std::string>::iterator itr = this -> leaves.begin();
    for (; itr != this -> leaves.end(); ++itr){lf[itr -> first] = _type + itr -> second;}
    this -> leaves = lf; 
}


void particle_template::build(std::map<std::string, particle_template*>* prt, element_t* el){
    return; 
}

particle_template* particle_template::clone(){return new particle_template();}
