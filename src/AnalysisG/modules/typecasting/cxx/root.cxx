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


