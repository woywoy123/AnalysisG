#include <element.h>
#include <report.h>
#include <folds.h>

// -------------------------- If you were directed here, simply add the data type within this section ----------------- //
// also make sure to checkout the structs/include/structs/element.h file! 

void data_t::flush_buffer(){
    // ------------ (5.) Add the buffer flush -------------------- //
    switch (this -> type){
        case data_enum::vvf: this -> flush_buffer(&this -> r_vvf); return;  
        case data_enum::vvd: this -> flush_buffer(&this -> r_vvd); return; 
        case data_enum::vvl: this -> flush_buffer(&this -> r_vvl); return; 
        case data_enum::vvi: this -> flush_buffer(&this -> r_vvi); return; 
        case data_enum::vvb: this -> flush_buffer(&this -> r_vvb); return; 

        case data_enum::vl:  this -> flush_buffer(&this -> r_vl ); return; 
        case data_enum::vd:  this -> flush_buffer(&this -> r_vd ); return; 
        case data_enum::vf:  this -> flush_buffer(&this -> r_vf ); return; 
        case data_enum::vi:  this -> flush_buffer(&this -> r_vi ); return; 
        case data_enum::vc:  this -> flush_buffer(&this -> r_vc ); return; 
        case data_enum::vb:  this -> flush_buffer(&this -> r_vb ); return; 
       
        case data_enum::ull: this -> flush_buffer(&this -> r_ull); return; 
        case data_enum::ui:  this -> flush_buffer(&this -> r_ui);  return; 
        case data_enum::d:   this -> flush_buffer(&this -> r_d  ); return; 
        case data_enum::l:   this -> flush_buffer(&this -> r_l  ); return; 
        case data_enum::f:   this -> flush_buffer(&this -> r_f  ); return; 
        case data_enum::i:   this -> flush_buffer(&this -> r_i  ); return; 
        case data_enum::b:   this -> flush_buffer(&this -> r_b  ); return; 
        case data_enum::c:   this -> flush_buffer(&this -> r_c  ); return; 
        default: return; 
    }
}

void data_t::fetch_buffer(){
    // ------------ (6.) Add the fetch buffer -------------------- //
    switch (this -> type){
        case data_enum::vvf: return this -> fetch_buffer(&this -> r_vvf);
        case data_enum::vvd: return this -> fetch_buffer(&this -> r_vvd);
        case data_enum::vvl: return this -> fetch_buffer(&this -> r_vvl);
        case data_enum::vvi: return this -> fetch_buffer(&this -> r_vvi);
        case data_enum::vvb: return this -> fetch_buffer(&this -> r_vvb);

        case data_enum::vl:  return this -> fetch_buffer(&this -> r_vl );
        case data_enum::vd:  return this -> fetch_buffer(&this -> r_vd );
        case data_enum::vf:  return this -> fetch_buffer(&this -> r_vf );
        case data_enum::vi:  return this -> fetch_buffer(&this -> r_vi );
        case data_enum::vc:  return this -> fetch_buffer(&this -> r_vc );
        case data_enum::vb:  return this -> fetch_buffer(&this -> r_vb );

        case data_enum::ull: return this -> fetch_buffer(&this -> r_ull);
        case data_enum::ui:  return this -> fetch_buffer(&this -> r_ui );
        case data_enum::l:   return this -> fetch_buffer(&this -> r_l  );
        case data_enum::d:   return this -> fetch_buffer(&this -> r_d  );
        case data_enum::f:   return this -> fetch_buffer(&this -> r_f  );
        case data_enum::i:   return this -> fetch_buffer(&this -> r_i  );
        case data_enum::b:   return this -> fetch_buffer(&this -> r_b  );
        case data_enum::c:   return this -> fetch_buffer(&this -> r_c  );
        default: return; 
    }
    // -> go to core/structs.pxd
}

void data_t::string_type(){

    // -------------------- (0). add the routing -------------- //
    if (this -> leaf_type == "vector<vector<float> >" ){ this -> type = data_enum::vvf; return;}
    if (this -> leaf_type == "vector<vector<double> >"){this -> type = data_enum::vvd; return;}
    if (this -> leaf_type == "vector<vector<long> >"  ){  this -> type = data_enum::vvl; return;}
    if (this -> leaf_type == "vector<vector<int> >"   ){   this -> type = data_enum::vvi; return;}
    if (this -> leaf_type == "vector<vector<bool> >"  ){  this -> type = data_enum::vvb; return;}

    if (this -> leaf_type == "vector<float>" ){ this -> type = data_enum::vf; return;}
    if (this -> leaf_type == "vector<long>"  ){  this -> type = data_enum::vl; return;}
    if (this -> leaf_type == "vector<int>"   ){   this -> type = data_enum::vi; return;}
    if (this -> leaf_type == "vector<char>"  ){  this -> type = data_enum::vc; return;}
    if (this -> leaf_type == "vector<bool>"  ){  this -> type = data_enum::vb; return;}
    if (this -> leaf_type == "vector<double>"){this -> type = data_enum::vd; return;}

    if (this -> leaf_type == "double" ){this -> type = data_enum::d;   return;}
    if (this -> leaf_type == "Float_t"){this -> type = data_enum::f;   return;}
    if (this -> leaf_type == "Int_t"  ){this -> type = data_enum::i;   return;}
    if (this -> leaf_type == "UInt_t" ){this -> type = data_enum::ui;  return;}
    if (this -> leaf_type == "Char_t" ){this -> type = data_enum::c;   return;}
    if (this -> leaf_type == "ULong64_t"){this -> type = data_enum::ull; return;}

    std::cout << "UNKNOWN TYPE: " << this -> leaf_type << " " << path << std::endl; 
    std::cout << "Add the type under modules/structs/cxx/structs.cxx" << std::endl;
    abort(); 
    // open -> /modules/structs/include/structs/element.h
}


// -------------- (4). add the data type interace ---------- //
bool data_t::element(std::vector<std::vector<float>>* el){
    if (!this -> r_vvf){return false;}
    (*el) = (*this -> r_vvf)[this -> index]; 
    return true; 
}

bool data_t::element(std::vector<std::vector<double>>* el){
    if (!this -> r_vvd){return false;}
    (*el) = (*this -> r_vvd)[this -> index]; 
    return true; 
}

bool data_t::element(std::vector<std::vector<long>>* el){
    if (!this -> r_vvl){return false;}
    (*el) = (*this -> r_vvl)[this -> index];
    return true; 
}

bool data_t::element(std::vector<std::vector<int>>* el){
    if (!this -> r_vvi){return false;} 
    (*el) = (*this -> r_vvi)[this -> index];
    return true; 
}

bool data_t::element(std::vector<std::vector<bool>>* el){
    if (!this -> r_vvb){return false;} 
    (*el) = (*this -> r_vvb)[this -> index];
    return true; 
}

bool data_t::element(std::vector<long>* el){
    if (!this -> r_vl){return false;}
    (*el) = (*this -> r_vl)[this -> index]; 
    return true; 
}

bool data_t::element(std::vector<double>* el){
    if (!this -> r_vd){return false;}
    (*el) = (*this -> r_vd)[this -> index]; 
    return true; 
}

bool data_t::element(std::vector<float>* el){
    if (!this -> r_vf){return false;}
    (*el) = (*this -> r_vf)[this -> index]; 
    return true; 
}

bool data_t::element(std::vector<int>* el){
    if (!this -> r_vi){return false;}
    (*el) = (*this -> r_vi)[this -> index]; 
    return true; 
}

bool data_t::element(std::vector<bool>* el){
    if (!this -> r_vb){return false;}
    (*el) = (*this -> r_vb)[this -> index]; 
    return true; 
}

bool data_t::element(std::vector<char>* el){
    if (!this -> r_vc){return false;}
    (*el) = (*this -> r_vc)[this -> index]; 
    return true; 
}

bool data_t::element(bool* el){
    if (!this -> r_b){return false;}
    (*el) = (*this -> r_b)[this -> index];
    return true; 
}

bool data_t::element(double* el){
    if (!this -> r_d){return false;}
    (*el) = (*this -> r_d)[this -> index];
    return true; 
}

bool data_t::element(float* el){
    if (!this -> r_f){return false;}
    (*el) = (*this -> r_f)[this -> index];
    return true; 
}

bool data_t::element(int* el){
    if (!this -> r_i){return false;}
    (*el) = (*this -> r_i)[this -> index];
    return true; 
}

bool data_t::element(long* el){
    if (!this -> r_l){return false;}
    (*el) = (*this -> r_l)[this -> index];
    return true; 
}

bool data_t::element(unsigned long long* el){
    if (!this -> r_ull){return false;}
    (*el) = (*this -> r_ull)[this -> index];
    return true; 
}

bool data_t::element(unsigned int* el){
    if (!this -> r_ui){return false;}
    (*el) = (*this -> r_ui)[this -> index];
    return true; 
}

bool data_t::element(char* el){
    if (!this -> r_c){return false;}
    (*el) = (*this -> r_c)[this -> index];
    return true; 
}

// ******************************************************************************************* //

