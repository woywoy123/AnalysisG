#include <vector_cast.h>


void add_to_dict(std::vector<std::vector<float>>*){gInterpreter -> GenerateDictionary("vector<vector<float>>", "vector");}
void add_to_dict(std::vector<std::vector<double>>*){gInterpreter -> GenerateDictionary("vector<vector<double>>", "vector");}
void add_to_dict(std::vector<std::vector<long>>*){gInterpreter -> GenerateDictionary("vector<vector<long>>", "vector");}
void add_to_dict(std::vector<std::vector<int>>*){gInterpreter -> GenerateDictionary("vector<vector<int>>", "vector");}
void add_to_dict(std::vector<std::vector<bool>>*){gInterpreter -> GenerateDictionary("vector<vector<bool>>", "vector");}

void add_to_dict(std::vector<float>*){gInterpreter -> GenerateDictionary("vector<float>", "vector");}
void add_to_dict(std::vector<double>*){gInterpreter -> GenerateDictionary("vector<double>", "vector");}
void add_to_dict(std::vector<long>*){gInterpreter -> GenerateDictionary("vector<long>", "vector");}
void add_to_dict(std::vector<int>*){gInterpreter -> GenerateDictionary("vector<int>", "vector");}
void add_to_dict(std::vector<bool>*){gInterpreter -> GenerateDictionary("vector<bool>", "vector");}

void add_to_dict(float*){gInterpreter -> GenerateDictionary("float"  , "vector");}
void add_to_dict(double*){gInterpreter -> GenerateDictionary("double", "vector");}
void add_to_dict(long*){gInterpreter -> GenerateDictionary("long"    , "vector");}
void add_to_dict(int*){gInterpreter -> GenerateDictionary("int"      , "vector");}
void add_to_dict(bool*){gInterpreter -> GenerateDictionary("bool"    , "vector");}


std::vector<signed long> tensor_size(torch::Tensor* inpt){
    c10::IntArrayRef dims = inpt -> sizes();
    std::vector<signed long> out; 
    for (size_t x(0); x < dims.size(); ++x){out.push_back(dims[x]);}
    return out;  
}

void variable_t::process(torch::Tensor* data, std::string* varname, TTree* tr){
    if (!this -> tt){this -> tt = tr; this -> variable_name = *varname;}
    std::vector<signed long> s = tensor_size(data); 

    // type and dim switch for the tensors
    if (s.size() == 2 && data -> dtype() == torch::kDouble){
        return this -> add_data(&this -> vvd, data, &s, varname, double(0));
    }

    if (s.size() == 1 && data -> dtype() == torch::kDouble){
        return this -> add_data(&this -> vd, data, &s, varname, double(0));
    }

    if (s.size() == 2 && data -> dtype() == torch::kFloat32){
        return this -> add_data(&this -> vvf, data, &s, varname, float(0));
    }

    if (s.size() == 1 && data -> dtype() == torch::kFloat32){
        return this -> add_data(&this -> vf, data, &s, varname, float(0));
    }

    if (s.size() == 2 && data -> dtype() == torch::kLong){
        return this -> add_data(&this -> vvl, data, &s, varname, long(0)); 
    }

    if (s.size() == 1 && data -> dtype() == torch::kLong){
        return this -> add_data(&this -> vl, data, &s, varname, long(0)); 
    }

    if (s.size() == 2 && data -> dtype() == torch::kInt){
        return this -> add_data(&this -> vvi, data, &s, varname, int(0)); 
    }

    if (s.size() == 1 && data -> dtype() == torch::kInt){
        return this -> add_data(&this -> vi, data, &s, varname, int(0)); 
    }

    if (s.size() == 2 && data -> dtype() == torch::kInt){
        return this -> add_data(&this -> vvi, data, &s, varname, int(0)); 
    }

    if (s.size() == 1 && data -> dtype() == torch::kBool){
        return this -> add_data(&this -> vb, data, &s, varname, bool(0)); 
    }

    if (s.size() == 2 && data -> dtype() == torch::kBool){
        return this -> add_data(&this -> vvb, data, &s, varname, bool(0)); 
    }


    std::cout << "DIM: " << s.size() << std::endl;
    std::cout << "Tensor Type: " << data -> dtype() << std::endl; 
    std::cout << *data << std::endl; 
    std::cout << "UNDEFINED DATA TYPE! SEE typecasting/cxx/typecasting.cxx" << std::endl;
    abort(); 
}



void variable_t::process(std::vector<std::vector<float>>* data, std::string* varname, TTree* tr){
    if (!this -> tt){this -> tt = tr; this -> variable_name = *varname;}
    this -> add_data(data, &this -> vvf); 
}

void variable_t::process(std::vector<std::vector<double>>* data, std::string* varname, TTree* tr){
    if (!this -> tt){this -> tt = tr; this -> variable_name = *varname;}
    this -> add_data(data, &this -> vvd); 
}

void variable_t::process(std::vector<std::vector<long>>* data, std::string* varname, TTree* tr){
    if (!this -> tt){this -> tt = tr; this -> variable_name = *varname;}
    this -> add_data(data, &this -> vvl); 
}

void variable_t::process(std::vector<std::vector<int>>* data, std::string* varname, TTree* tr){
    if (!this -> tt){this -> tt = tr; this -> variable_name = *varname;}
    this -> add_data(data, &this -> vvi); 
}

void variable_t::process(std::vector<std::vector<bool>>* data, std::string* varname, TTree* tr){
    if (!this -> tt){this -> tt = tr; this -> variable_name = *varname;}
    this -> add_data(data, &this -> vvb); 
}

void variable_t::process(std::vector<float>* data, std::string* varname, TTree* tr){
    if (!this -> tt){this -> tt = tr; this -> variable_name = *varname;}
    this -> add_data(data, &this -> vf); 
}

void variable_t::process(std::vector<double>* data, std::string* varname, TTree* tr){
    if (!this -> tt){this -> tt = tr; this -> variable_name = *varname;}
    this -> add_data(data, &this -> vd); 
}

void variable_t::process(std::vector<long>* data, std::string* varname, TTree* tr){
    if (!this -> tt){this -> tt = tr; this -> variable_name = *varname;}
    this -> add_data(data, &this -> vl); 
}

void variable_t::process(std::vector<int>* data, std::string* varname, TTree* tr){
    if (!this -> tt){this -> tt = tr; this -> variable_name = *varname;}
    this -> add_data(data, &this -> vi); 
}

void variable_t::process(std::vector<bool>* data, std::string* varname, TTree* tr){
    if (!this -> tt){this -> tt = tr; this -> variable_name = *varname;}
    this -> add_data(data, &this -> vb); 
}


void variable_t::process(float* data, std::string* varname, TTree* tr){
    if (!this -> tt){this -> tt = tr; this -> variable_name = *varname;}
    this -> add_data(data, &this -> f); 
}

void variable_t::process(double* data, std::string* varname, TTree* tr){
    if (!this -> tt){this -> tt = tr; this -> variable_name = *varname;}
    this -> add_data(data, &this -> d); 
}

void variable_t::process(long* data, std::string* varname, TTree* tr){
    if (!this -> tt){this -> tt = tr; this -> variable_name = *varname;}
    this -> add_data(data, &this -> l); 
}

void variable_t::process(int* data, std::string* varname, TTree* tr){
    if (!this -> tt){this -> tt = tr; this -> variable_name = *varname;}
    this -> add_data(data, &this -> i); 
}

void variable_t::process(bool* data, std::string* varname, TTree* tr){
    if (!this -> tt){this -> tt = tr; this -> variable_name = *varname;}
    this -> add_data(data, &this -> b); 
}

















void variable_t::flush(){
    if (this -> vvf.size()){this -> vvf.clear(); return; }
    if (this -> vvd.size()){this -> vvd.clear(); return; }
    if (this -> vvl.size()){this -> vvl.clear(); return; }
    if (this -> vvi.size()){this -> vvi.clear(); return; }
    if (this -> vvb.size()){this -> vvb.clear(); return; }

    if (this -> vf.size()){this -> vf.clear(); return; }
    if (this -> vd.size()){this -> vd.clear(); return; }
    if (this -> vl.size()){this -> vl.clear(); return; }
    if (this -> vi.size()){this -> vi.clear(); return; }
    if (this -> vb.size()){this -> vb.clear(); return; }
}
