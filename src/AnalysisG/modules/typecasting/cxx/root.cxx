#include <vector_cast.h>
#include <string.h>

void static buildDict(std::string _name, std::string _shrt){
    std::string name = std::string(_name);
    std::string shrt = std::string(_shrt); 
    //if (gInterpreter -> IsLoaded(shrt.c_str())){return;}
    gInterpreter -> GenerateDictionary(name.c_str(), shrt.c_str()); 
}

void add_to_dict(std::vector<std::vector<float>>*){buildDict("vector<vector<float>>", "vector");}
void add_to_dict(std::vector<std::vector<double>>*){buildDict("vector<vector<double>>", "vector");}
void add_to_dict(std::vector<std::vector<long>>*){buildDict("vector<vector<long>>", "vector");}
void add_to_dict(std::vector<std::vector<int>>*){buildDict("vector<vector<int>>", "vector");}
void add_to_dict(std::vector<std::vector<bool>>*){buildDict("vector<vector<bool>>", "vector");}

void add_to_dict(std::vector<float>*){buildDict("vector<float>", "vector");}
void add_to_dict(std::vector<double>*){buildDict("vector<double>", "vector");}
void add_to_dict(std::vector<long>*){buildDict("vector<long>", "vector");}
void add_to_dict(std::vector<int>*){buildDict("vector<int>", "vector");}
void add_to_dict(std::vector<bool>*){buildDict("vector<bool>", "vector");}

void add_to_dict(float*){buildDict("float"  , "vector");}
void add_to_dict(double*){buildDict("double", "vector");}
void add_to_dict(long*){buildDict("long"    , "vector");}
void add_to_dict(int*){buildDict("int"      , "vector");}
void add_to_dict(bool*){buildDict("bool"    , "vector");}


void variable_t::process(std::vector<std::vector<float>>* data, std::string* varname, TTree* tr){
    this -> add_data(data, &this -> vvf, varname, tr);
}

void variable_t::process(std::vector<std::vector<double>>* data, std::string* varname, TTree* tr){
    this -> add_data(data, &this -> vvd, varname, tr); 
}

void variable_t::process(std::vector<std::vector<long>>* data, std::string* varname, TTree* tr){
    this -> add_data(data, &this -> vvl, varname, tr); 
}

void variable_t::process(std::vector<std::vector<int>>* data, std::string* varname, TTree* tr){
    this -> add_data(data, &this -> vvi, varname, tr); 
}

void variable_t::process(std::vector<std::vector<bool>>* data, std::string* varname, TTree* tr){
    this -> add_data(data, &this -> vvb, varname, tr); 
}

void variable_t::process(std::vector<float>* data, std::string* varname, TTree* tr){
    this -> add_data(data, &this -> vf, varname, tr); 
}

void variable_t::process(std::vector<double>* data, std::string* varname, TTree* tr){
    this -> add_data(data, &this -> vd, varname, tr); 
}

void variable_t::process(std::vector<long>* data, std::string* varname, TTree* tr){
    this -> add_data(data, &this -> vl, varname, tr); 
}

void variable_t::process(std::vector<int>* data, std::string* varname, TTree* tr){
    this -> add_data(data, &this -> vi, varname, tr); 
}

void variable_t::process(std::vector<bool>* data, std::string* varname, TTree* tr){
    this -> add_data(data, &this -> vb, varname, tr); 
}


void variable_t::process(float* data, std::string* varname, TTree* tr){
    this -> add_data(data, &this -> f, varname, tr); 
}

void variable_t::process(double* data, std::string* varname, TTree* tr){
    this -> add_data(data, &this -> d, varname, tr); 
}

void variable_t::process(long* data, std::string* varname, TTree* tr){
    this -> add_data(data, &this -> l, varname, tr); 
}

void variable_t::process(int* data, std::string* varname, TTree* tr){
    this -> add_data(data, &this -> i, varname, tr); 
}

void variable_t::process(bool* data, std::string* varname, TTree* tr){
    this -> add_data(data, &this -> b, varname, tr); 
}


