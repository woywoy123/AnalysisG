#include <templates/selection_template.h>
#include <meta/meta.h>

bool selection_template::selection(event_template*){return true;}
bool selection_template::strategy(event_template*){return true;}
selection_template* selection_template::clone(){return new selection_template();}
void selection_template::merge(selection_template*){}
void selection_template::bulk_write(const long*, std::string*){this -> p_bulk_write = false;}

void selection_template::write(std::vector<particle_template*>* particles, std::string name, particle_enum attrs){
    std::vector<int> data_i = {}; 
    std::vector<bool> data_b = {}; 
    std::vector<double> data_d = {}; 
    std::vector<std::vector<double>> data_dd = {}; 

    data_enum types = data_enum::undef; 
    switch (attrs){
        case particle_enum::pt:     name += "_pt"    ; types = data_enum::v_d;  break;   
        case particle_enum::eta:    name += "_eta"   ; types = data_enum::v_d;  break; 
        case particle_enum::phi:    name += "_phi"   ; types = data_enum::v_d;  break; 
        case particle_enum::px:     name += "_px"    ; types = data_enum::v_d;  break; 
        case particle_enum::py:     name += "_py"    ; types = data_enum::v_d;  break; 
        case particle_enum::pz:     name += "_pz"    ; types = data_enum::v_d;  break; 
        case particle_enum::mass:   name += "_mass"  ; types = data_enum::v_d;  break; 
        case particle_enum::energy: name += "_energy"; types = data_enum::v_d;  break; 
        case particle_enum::charge: name += "_charge"; types = data_enum::v_d;  break; 
        case particle_enum::is_b:   name += "_is_b"  ; types = data_enum::v_b;  break; 
        case particle_enum::is_lep: name += "_is_l"  ; types = data_enum::v_b;  break; 
        case particle_enum::is_nu:  name += "_is_n"  ; types = data_enum::v_b;  break; 
        case particle_enum::is_add: name += "_is_a"  ; types = data_enum::v_b;  break; 
        case particle_enum::pdgid:  name += "_pdgid" ; types = data_enum::v_i;  break; 
        case particle_enum::index:  name += "_index" ; types = data_enum::v_i;  break; 
        case particle_enum::pmc:    name += "_pmc"   ; types = data_enum::vv_d; break; 
        case particle_enum::pmu:    name += "_pmu"   ; types = data_enum::vv_d; break; 
        default: return; 
    }

    for (size_t x(0); x < particles -> size(); ++x){
        switch (types){
            case data_enum::v_d:  this -> switch_board(attrs, particles -> at(x), &data_d);  break;  
            case data_enum::v_i:  this -> switch_board(attrs, particles -> at(x), &data_i);  break;  
            case data_enum::v_b:  this -> switch_board(attrs, particles -> at(x), &data_b);  break;  
            case data_enum::vv_d: this -> switch_board(attrs, particles -> at(x), &data_dd); break; 
            default: break; 
        }
    }

    switch (types){
        case data_enum::v_d:  return this -> write(&data_d,  name);
        case data_enum::v_i:  return this -> write(&data_i,  name);
        case data_enum::v_b:  return this -> write(&data_b,  name);
        case data_enum::vv_d: return this -> write(&data_dd, name);
        default: return; 
    }
}
