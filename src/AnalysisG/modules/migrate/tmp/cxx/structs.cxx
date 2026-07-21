#include <element.h>
#include <report.h>
#include <folds.h>

// -------------------------- If you were directed here, simply add the data type within this section ----------------- //
// also make sure to checkout the structs/include/structs/element.h file! 

void data_t::fetch_buffer(){
    // ------------ (6.) Add the fetch buffer -------------------- //
    switch (this -> type){
        case data_enum::vvv_ull:  this -> fetch_buffer(&this -> vvv_ull); return; 
        case data_enum::vvv_ui:   this -> fetch_buffer(&this -> vvv_ui ); return; 
        case data_enum::vvv_d:    this -> fetch_buffer(&this -> vvv_d  ); return; 
        case data_enum::vvv_l:    this -> fetch_buffer(&this -> vvv_l  ); return; 
        case data_enum::vvv_f:    this -> fetch_buffer(&this -> vvv_f  ); return; 
        case data_enum::vvv_i:    this -> fetch_buffer(&this -> vvv_i  ); return; 
        case data_enum::vvv_b:    this -> fetch_buffer(&this -> vvv_b  ); return; 
        case data_enum::vvv_c:    this -> fetch_buffer(&this -> vvv_c  ); return; 
        
        case data_enum::vv_ull:   this -> fetch_buffer(&this -> vv_ull ); return; 
        case data_enum::vv_ui:    this -> fetch_buffer(&this -> vv_ui  ); return; 
        case data_enum::vv_d:     this -> fetch_buffer(&this -> vv_d   ); return; 
        case data_enum::vv_l:     this -> fetch_buffer(&this -> vv_l   ); return; 
        case data_enum::vv_f:     this -> fetch_buffer(&this -> vv_f   ); return; 
        case data_enum::vv_i:     this -> fetch_buffer(&this -> vv_i   ); return; 
        case data_enum::vv_b:     this -> fetch_buffer(&this -> vv_b   ); return; 
        case data_enum::vv_c:     this -> fetch_buffer(&this -> vv_c   ); return; 
        
        case data_enum::v_ull:    this -> fetch_buffer(&this -> v_ull  ); return; 
        case data_enum::v_ui:     this -> fetch_buffer(&this -> v_ui   ); return; 
        case data_enum::v_d:      this -> fetch_buffer(&this -> v_d    ); return; 
        case data_enum::v_l:      this -> fetch_buffer(&this -> v_l    ); return; 
        case data_enum::v_f:      this -> fetch_buffer(&this -> v_f    ); return; 
        case data_enum::v_i:      this -> fetch_buffer(&this -> v_i    ); return; 
        case data_enum::v_b:      this -> fetch_buffer(&this -> v_b    ); return; 
        case data_enum::v_c:      this -> fetch_buffer(&this -> v_c    ); return; 
        default: return; 
    }
    // -> go to core/structs.pxd
}

void data_t::string_type(){
    this -> type = this -> root_type_translate(&this -> leaf_type); 
    if (this -> type != data_enum::undef){return;}
    std::cout << "UNKNOWN TYPE: " << this -> leaf_type << " " << path << std::endl; 
    std::cout << "Add the type under -> /modules/structs/include/structs/base.h" << std::endl;
    abort(); 
}


