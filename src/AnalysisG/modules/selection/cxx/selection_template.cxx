#include <templates/selection_template.h>
#include <meta/meta.h>

bool selection_template::selection(event_template*){return true;}
bool selection_template::strategy(event_template*){return true;}
selection_template* selection_template::clone(){return new selection_template();}
void selection_template::merge(selection_template*){}
void selection_template::bulk_write(const long*, std::string*){this -> p_bulk_write = false;}

void selection_template::write(std::vector<particle_template*>* particles, std::string _name, particle_enum attrs){
    std::vector<int> data_i = {}; 
    std::vector<bool> data_b = {}; 
    std::vector<double> data_d = {}; 
    std::vector<std::vector<double>> data_dd = {}; 

    data_enum types = data_enum::undef; 
    switch (attrs){
        case particle_enum::pt:     _name += "_pt"    ; types = data_enum::v_d;  break;   
        case particle_enum::eta:    _name += "_eta"   ; types = data_enum::v_d;  break; 
        case particle_enum::phi:    _name += "_phi"   ; types = data_enum::v_d;  break; 
        case particle_enum::px:     _name += "_px"    ; types = data_enum::v_d;  break; 
        case particle_enum::py:     _name += "_py"    ; types = data_enum::v_d;  break; 
        case particle_enum::pz:     _name += "_pz"    ; types = data_enum::v_d;  break; 
        case particle_enum::mass:   _name += "_mass"  ; types = data_enum::v_d;  break; 
        case particle_enum::energy: _name += "_energy"; types = data_enum::v_d;  break; 
        case particle_enum::charge: _name += "_charge"; types = data_enum::v_d;  break; 
        case particle_enum::is_b:   _name += "_is_b"  ; types = data_enum::v_b;  break; 
        case particle_enum::is_lep: _name += "_is_l"  ; types = data_enum::v_b;  break; 
        case particle_enum::is_nu:  _name += "_is_n"  ; types = data_enum::v_b;  break; 
        case particle_enum::is_add: _name += "_is_a"  ; types = data_enum::v_b;  break; 
        case particle_enum::pdgid:  _name += "_pdgid" ; types = data_enum::v_i;  break; 
        case particle_enum::index:  _name += "_index" ; types = data_enum::v_i;  break; 
        case particle_enum::pmc:    _name += "_pmc"   ; types = data_enum::vv_d; break; 
        case particle_enum::pmu:    _name += "_pmu"   ; types = data_enum::vv_d; break; 
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
        case data_enum::v_d:  return this -> write(&data_d,  _name);
        case data_enum::v_i:  return this -> write(&data_i,  _name);
        case data_enum::v_b:  return this -> write(&data_b,  _name);
        case data_enum::vv_d: return this -> write(&data_dd, _name);
        default: return; 
    }
}
