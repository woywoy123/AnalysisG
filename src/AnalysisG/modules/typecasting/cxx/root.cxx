#include <tools/vector_cast.h>

void variable_t::process(std::vector<std::vector<float>>* data, std::string* varname, TTree* tr){
    this -> add_data(data, this -> vv_f, varname, tr);
}

void variable_t::process(std::vector<std::vector<double>>* data, std::string* varname, TTree* tr){
    this -> add_data(data, this -> vv_d, varname, tr); 
}

void variable_t::process(std::vector<std::vector<long>>* data, std::string* varname, TTree* tr){
    this -> add_data(data, this -> vv_l, varname, tr); 
}

void variable_t::process(std::vector<std::vector<int>>* data, std::string* varname, TTree* tr){
    this -> add_data(data, this -> vv_i, varname, tr); 
}

void variable_t::process(std::vector<std::vector<bool>>* data, std::string* varname, TTree* tr){
    this -> add_data(data, this -> vv_b, varname, tr); 
}

void variable_t::process(std::vector<float>* data, std::string* varname, TTree* tr){
    this -> add_data(data, this -> v_f, varname, tr); 
}

void variable_t::process(std::vector<double>* data, std::string* varname, TTree* tr){
    this -> add_data(data, this -> v_d, varname, tr); 
}

void variable_t::process(std::vector<long>* data, std::string* varname, TTree* tr){
    this -> add_data(data, this -> v_l, varname, tr); 
}

void variable_t::process(std::vector<int>* data, std::string* varname, TTree* tr){
    this -> add_data(data, this -> v_i, varname, tr); 
}

void variable_t::process(std::vector<bool>* data, std::string* varname, TTree* tr){
    this -> add_data(data, this -> v_b, varname, tr); 
}

void variable_t::process(float* data, std::string* varname, TTree* tr){
    this -> add_data(data, this -> f, varname, tr); 
}

void variable_t::process(double* data, std::string* varname, TTree* tr){
    this -> add_data(data, this -> d, varname, tr); 
}

void variable_t::process(long* data, std::string* varname, TTree* tr){
    this -> add_data(data, this -> l, varname, tr); 
}

void variable_t::process(int* data, std::string* varname, TTree* tr){
    this -> add_data(data, this -> i, varname, tr); 
}

void variable_t::process(bool* data, std::string* varname, TTree* tr){
    this -> add_data(data, this -> b, varname, tr); 
}
// ==================================================================================== //

