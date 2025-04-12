#include <vector_cast.h>

std::vector<signed long> tensor_size(torch::Tensor* inpt){
    c10::IntArrayRef dims = inpt -> sizes();
    std::vector<signed long> out; 
    for (size_t x(0); x < dims.size(); ++x){out.push_back(dims[x]);}
    return out;  
}

variable_t::variable_t(){}
variable_t::variable_t(bool ux){ this -> use_external = ux;}
variable_t::~variable_t(){
    if (this -> type == data_enum::unset){return;}
    this -> clear = true; 
    this -> flush_buffer(); 
}

void variable_t::create_meta(meta_t* mtf){
    if (mtf){this -> mtx = mtf;}
    if (!this -> mtx){return;}
    this -> tt = new TTree("MetaData", "meta"); 
    this -> tt -> Branch("MetaData", mtf); 
    this -> tt -> Fill(); 
    this -> tt -> Write("", TObject::kOverwrite);  
    delete this -> tt; 
    this -> tt = nullptr; 
    this -> mtx = nullptr; 
}

void variable_t::build_switch(size_t s, torch::Tensor* tx){
    // ======================= DEFINE YOUR VARIABLES HERE!!! (1) ========================= //
    // type and dim switch for the tensors

    // ---------- Two dimensional matrices ------------ 
    if      (s == 3 && tx -> dtype() == torch::kDouble ){this -> type = data_enum::vvv_d;}
    else if (s == 3 && tx -> dtype() == torch::kFloat32){this -> type = data_enum::vvv_f;}
    else if (s == 3 && tx -> dtype() == torch::kLong   ){this -> type = data_enum::vvv_l;}   
    else if (s == 3 && tx -> dtype() == torch::kInt    ){this -> type = data_enum::vvv_i;}    
    else if (s == 3 && tx -> dtype() == torch::kBool   ){this -> type = data_enum::vvv_b;}   

    // ---------- Two dimensional matrices ------------ 
    else if (s == 2 && tx -> dtype() == torch::kDouble ){this -> type = data_enum::vv_d;}
    else if (s == 2 && tx -> dtype() == torch::kFloat32){this -> type = data_enum::vv_f;}
    else if (s == 2 && tx -> dtype() == torch::kLong   ){this -> type = data_enum::vv_l;}   
    else if (s == 2 && tx -> dtype() == torch::kInt    ){this -> type = data_enum::vv_i;}    
    else if (s == 2 && tx -> dtype() == torch::kBool   ){this -> type = data_enum::vv_b;}   

    // ---------- One dimensional matrices ------------ 
    else if (s == 1 && tx -> dtype() == torch::kDouble ){this -> type = data_enum::v_d; } 
    else if (s == 1 && tx -> dtype() == torch::kFloat32){this -> type = data_enum::v_f; }
    else if (s == 1 && tx -> dtype() == torch::kLong   ){this -> type = data_enum::v_l; }   
    else if (s == 1 && tx -> dtype() == torch::kInt    ){this -> type = data_enum::v_i; }    
    else if (s == 1 && tx -> dtype() == torch::kBool   ){this -> type = data_enum::v_b; }    
    else {this -> type = data_enum::undef;}
    if (this -> type != data_enum::undef){return;}

    std::cout << "DIM: " << s << std::endl;
    std::cout << "Tensor Type: " << tx -> dtype() << std::endl; 
    std::cout << *tx << std::endl; 
    std::cout << "UNDEFINED DATA TYPE! SEE typecasting/cxx/typecasting.cxx" << std::endl;
    abort(); 

    // then go to modules/structs/base.h -> data_enum and add your type.
    // =================================================================================== //
}

void variable_t::process(torch::Tensor* data, std::string* varname, TTree* tr){
    std::vector<signed long> s = tensor_size(data); 
    if (this -> type == data_enum::unset && varname){
        this -> build_switch(s.size(), data);
        this -> variable_name = *varname;
    }
    if (!this -> tt && tr){
        this -> create_meta(nullptr);
        this -> tt = tr;
    }
    this -> flush_buffer();

    // ============================== Add your type (5) ==================================== //
    switch(this -> type){
        case data_enum::vv_d: this -> add_data(this -> vv_d, data, &s, double(0)); break;  
        case data_enum::vv_f: this -> add_data(this -> vv_f, data, &s, float(0) ); break;  
        case data_enum::vv_l: this -> add_data(this -> vv_l, data, &s, long(0)  ); break; 
        case data_enum::vv_i: this -> add_data(this -> vv_i, data, &s, int(0)   ); break; 
        case data_enum::vv_b: this -> add_data(this -> vv_b, data, &s, bool(0)  ); break; 
        case data_enum::v_d:  this -> add_data(this ->  v_d, data, &s, double(0)); break; 
        case data_enum::v_f:  this -> add_data(this ->  v_f, data, &s, float(0) ); break; 
        case data_enum::v_l:  this -> add_data(this ->  v_l, data, &s, long(0)  ); break; 
        case data_enum::v_i:  this -> add_data(this ->  v_i, data, &s, int(0)   ); break; 
        case data_enum::v_b:  this -> add_data(this ->  v_b, data, &s, bool(0)  ); break; 
        default: break; 
    }
    // ===================================================================================== //
    if (!this -> tt || !this -> tb || !this -> is_triggered){return;}
    if (this -> is_triggered){this -> tt -> AddBranchToCache(this -> tb, true);}
    this -> is_triggered = false; 
}


