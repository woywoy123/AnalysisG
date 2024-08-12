#include <element.h>

void element_t::set_meta(){
    std::map<std::string, data_t*>::iterator itr = this -> handle.begin();
    bool sk = itr -> second -> file_index >= (int)itr -> second -> files_i -> size(); 
    if (sk){return;}
    this -> event_index = itr -> second -> index; 
    this -> filename = itr -> second -> files_s -> at(itr -> second -> file_index);
}

bool element_t::next(){
    bool stop = true; 
    std::map<std::string, data_t*>::iterator itr = this -> handle.begin(); 
    for (; itr != this -> handle.end(); ++itr){stop *= itr -> second -> next();}
    return stop; 
}

// -------------------------- If you were directed here, simply add the data type within this section ----------------- //

void data_t::flush_buffer(){
    switch (this -> type){
        case data_enum::vvf: this -> flush_buffer(&this -> r_vvf); return; 
        case data_enum::vvd: this -> flush_buffer(&this -> r_vvd); return; 
        case data_enum::vvl: this -> flush_buffer(&this -> r_vvl); return; 
        case data_enum::vvi: this -> flush_buffer(&this -> r_vvi); return; 

        case data_enum::vf:  this -> flush_buffer(&this -> r_vf ); return; 
        case data_enum::vl:  this -> flush_buffer(&this -> r_vl ); return; 
        case data_enum::vi:  this -> flush_buffer(&this -> r_vi ); return; 
        case data_enum::vc:  this -> flush_buffer(&this -> r_vc ); return; 
        case data_enum::vb:  this -> flush_buffer(&this -> r_vb ); return; 

        case data_enum::f:   this -> flush_buffer(&this -> r_f  ); return; 
        case data_enum::l:   this -> flush_buffer(&this -> r_l  ); return; 
        case data_enum::i:   this -> flush_buffer(&this -> r_i  ); return; 
        case data_enum::b:   this -> flush_buffer(&this -> r_b  ); return; 
        case data_enum::ull: this -> flush_buffer(&this -> r_ull); return; 
        default: return; 
    }
}

void data_t::fetch_buffer(){
    switch (this -> type){
        case data_enum::vvf: return this -> fetch_buffer(&this -> r_vvf);
        case data_enum::vvd: return this -> fetch_buffer(&this -> r_vvd);
        case data_enum::vvl: return this -> fetch_buffer(&this -> r_vvl);
        case data_enum::vvi: return this -> fetch_buffer(&this -> r_vvi);

        case data_enum::vf:  return this -> fetch_buffer(&this -> r_vf );
        case data_enum::vl:  return this -> fetch_buffer(&this -> r_vl );
        case data_enum::vi:  return this -> fetch_buffer(&this -> r_vi );
        case data_enum::vc:  return this -> fetch_buffer(&this -> r_vc );
        case data_enum::vb:  return this -> fetch_buffer(&this -> r_vb );

        case data_enum::f:   return this -> fetch_buffer(&this -> r_f  );
        case data_enum::l:   return this -> fetch_buffer(&this -> r_l  );
        case data_enum::i:   return this -> fetch_buffer(&this -> r_i  );
        case data_enum::b:   return this -> fetch_buffer(&this -> r_b  );
        case data_enum::ull: return this -> fetch_buffer(&this -> r_ull);
        default: return; 
    }
}

void data_t::string_type(){

    if (this -> leaf_type == "vector<vector<float> >"){ this -> type = data_enum::vvf; return;}
    if (this -> leaf_type == "vector<vector<double> >"){this -> type = data_enum::vvd; return;}
    if (this -> leaf_type == "vector<vector<long> >"){  this -> type = data_enum::vvl; return;}
    if (this -> leaf_type == "vector<vector<int> >"){   this -> type = data_enum::vvi; return;}

    if (this -> leaf_type == "vector<float>"){this -> type = data_enum::vf; return;}
    if (this -> leaf_type == "vector<long>"){ this -> type = data_enum::vl; return;}
    if (this -> leaf_type == "vector<int>"){  this -> type = data_enum::vi; return;}
    if (this -> leaf_type == "vector<char>"){ this -> type = data_enum::vc; return;}
    if (this -> leaf_type == "vector<bool>"){ this -> type = data_enum::vb; return;}

    if (this -> leaf_type == "Float_t"){  this -> type = data_enum::f; return;}
    if (this -> leaf_type == "Int_t"){    this -> type = data_enum::i; return;}
    if (this -> leaf_type == "ULong64_t"){this -> type = data_enum::ull; return;}

    std::cout << "UNKNOWN TYPE: " << this -> leaf_type << " " << path << std::endl; 
    std::cout << "Add the type under modules/structs/cxx/structs.cxx" << std::endl;
    abort(); 
}

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

bool data_t::element(std::vector<long>* el){
    if (!this -> r_vl){return false;}
    (*el) = (*this -> r_vl)[this -> index]; 
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

// ******************************************************************************************* //

void data_t::flush(){
    this -> flush_buffer();
    if (this -> files_s){delete this -> files_s;}
    if (this -> files_i){delete this -> files_i;}
    if (this -> files_t){delete this -> files_t;}
}

void data_t::initialize(){
    TFile* c = this -> files_t -> at(this -> file_index); 
    c = c -> Open(c -> GetTitle()); 

    this -> tree        = (TTree*)c -> Get(this -> tree_name.c_str()); 
    this -> leaf        = this -> tree -> FindLeaf(this -> leaf_name.c_str());
    this -> branch      = this -> leaf -> GetBranch();  
    
    this -> tree_name   = this -> tree -> GetName();
    this -> leaf_name   = this -> leaf -> GetName();
    this -> branch_name = this -> branch -> GetName(); 

    this -> string_type(); 
    this -> flush_buffer(); 
    this -> fetch_buffer(); 
    this -> index = 0; 
    c -> Close(); 
    delete c; 
} 

bool data_t::next(){
    bool sk = this -> file_index >= (int)this -> files_i -> size(); 
    if (sk){return true;}

    long idx = this -> files_i -> at(this -> file_index);
    if (this -> index+1 < idx){this -> index++; return false;}

    this -> file_index++; 
    sk = this -> file_index >= (int)this -> files_i -> size(); 
    if (sk){return true;}
    this -> initialize();
    return false; 
}


