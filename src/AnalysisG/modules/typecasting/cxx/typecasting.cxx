#include <vector_cast.h>
#include <filesystem>

std::vector<signed long> tensor_size(torch::Tensor* inpt){
    c10::IntArrayRef dims = inpt -> sizes();
    std::vector<signed long> out; 
    for (size_t x(0); x < dims.size(); ++x){out.push_back(dims[x]);}
    return out;  
}

void variable_t::build_switch(size_t s, torch::Tensor* tx){
    if (this -> vr != var_enum::unset){return;}

    // type and dim switch for the tensors
    // ---------- Two dimensional matrices ------------ 
    if      (s == 2 && tx -> dtype() == torch::kDouble ){this -> vr = var_enum::vvd;}
    else if (s == 2 && tx -> dtype() == torch::kFloat32){this -> vr = var_enum::vvf;}
    else if (s == 2 && tx -> dtype() == torch::kLong   ){this -> vr = var_enum::vvl;}   
    else if (s == 2 && tx -> dtype() == torch::kInt    ){this -> vr = var_enum::vvi;}    
    else if (s == 2 && tx -> dtype() == torch::kBool   ){this -> vr = var_enum::vvb;}   

    // ---------- One dimensional matrices ------------ 
    else if (s == 1 && tx -> dtype() == torch::kDouble ){this -> vr = var_enum::vd ;} 
    else if (s == 1 && tx -> dtype() == torch::kFloat32){this -> vr = var_enum::vf ;}
    else if (s == 1 && tx -> dtype() == torch::kLong   ){this -> vr = var_enum::vl ;}   
    else if (s == 1 && tx -> dtype() == torch::kInt    ){this -> vr = var_enum::vi ;}    
    else if (s == 1 && tx -> dtype() == torch::kBool   ){this -> vr = var_enum::vb ;}    
    else {this -> vr = var_enum::undef;}
}

void variable_t::process(torch::Tensor* data, std::string* varname, TTree* tr){
    std::vector<signed long> s = tensor_size(data); 
    if (this -> vr == var_enum::unset && !varname){
        this -> build_switch(s.size(), data);
        this -> variable_name = *varname;
    }
    if (!this -> tt && tr){this -> tt = tr;}

    switch(this -> vr){
        case var_enum::vvd: return this -> add_data(&this -> vvd, data, &s, varname, double(0));
        case var_enum::vvf: return this -> add_data(&this -> vvf, data, &s, varname, float(0) );
        case var_enum::vvl: return this -> add_data(&this -> vvl, data, &s, varname, long(0)  );
        case var_enum::vvi: return this -> add_data(&this -> vvi, data, &s, varname, int(0)   );
        case var_enum::vvb: return this -> add_data(&this -> vvb, data, &s, varname, bool(0)  );
        case var_enum::vd : return this -> add_data(&this -> vd , data, &s, varname, double(0));
        case var_enum::vf : return this -> add_data(&this -> vf , data, &s, varname, float(0) );
        case var_enum::vl : return this -> add_data(&this -> vl , data, &s, varname, long(0)  );
        case var_enum::vi : return this -> add_data(&this -> vi , data, &s, varname, int(0)   );
        case var_enum::vb : return this -> add_data(&this -> vb , data, &s, varname, bool(0)  );
        default: break; 
    }
    std::cout << "DIM: " << s.size() << std::endl;
    std::cout << "Tensor Type: " << data -> dtype() << std::endl; 
    std::cout << *data << std::endl; 
    std::cout << "UNDEFINED DATA TYPE! SEE typecasting/cxx/typecasting.cxx" << std::endl;
    abort(); 
}

void variable_t::flush(){
    if (this -> vr != var_enum::unset){}
    else if (this -> vvf.size()){this -> vr = var_enum::vvf;}
    else if (this -> vvd.size()){this -> vr = var_enum::vvd;}
    else if (this -> vvl.size()){this -> vr = var_enum::vvl;}
    else if (this -> vvi.size()){this -> vr = var_enum::vvi;}
    else if (this -> vvb.size()){this -> vr = var_enum::vvb;}
    else if (this ->  vf.size()){this -> vr = var_enum::vf;}
    else if (this ->  vd.size()){this -> vr = var_enum::vd;}
    else if (this ->  vl.size()){this -> vr = var_enum::vl;}
    else if (this ->  vi.size()){this -> vr = var_enum::vi;}
    else if (this ->  vb.size()){this -> vr = var_enum::vb;}

    switch(this -> vr){
        case var_enum::vvd: this -> vvf.clear(); return;
        case var_enum::vvf: this -> vvd.clear(); return;
        case var_enum::vvl: this -> vvl.clear(); return;
        case var_enum::vvi: this -> vvi.clear(); return;
        case var_enum::vvb: this -> vvb.clear(); return;
        case var_enum::vd : this -> vf.clear();  return;
        case var_enum::vf : this -> vd.clear();  return;
        case var_enum::vl : this -> vl.clear();  return;
        case var_enum::vi : this -> vi.clear();  return;
        case var_enum::vb : this -> vb.clear();  return;
        default: break; 
    }
}
