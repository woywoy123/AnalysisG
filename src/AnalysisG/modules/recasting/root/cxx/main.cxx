#include <typecasting/elements.h>

//bool AnalysisG::core::basis_t::element(std::vector<std::vector<std::vector<float>>>* el){
//    return this -> _getalt(this -> vvv_f, el); 
//}
//
//bool AnalysiG::core::basis_t::element(std::vector<std::vector<std::vector<double>>>* el){
//    return this -> _getalt(this -> vvv_d, el); 
//}
//
//bool AnalysiG::core::basis_t::element(std::vector<std::vector<std::vector<long>>>* el){
//    return this -> _getalt(this -> vvv_l, el); 
//}
//
//bool AnalysiG::core::basis_t::element(std::vector<std::vector<std::vector<int>>>* el){
//    return this -> _getalt(this -> vvv_i, el); 
//}
//
//bool AnalysiG::core::basis_t::element(std::vector<std::vector<std::vector<bool>>>* el){
//    return this -> _getalt(this -> vvv_b, el); 
//}
//
//
//bool AnalysiG::core::basis_t::element(std::vector<std::vector<float>>* el){
//    return this -> _getalt(this -> vvv_f, this -> vv_f, el); 
//}
//
//bool AnalysiG::core::basis_t::element(std::vector<std::vector<double>>* el){
//    return this -> _getalt(this -> vvv_d, this -> vv_d, el); 
//}
//
//bool AnalysiG::core::basis_t::element(std::vector<std::vector<long>>* el){
//    return this -> _getalt(this -> vvv_l, this -> vv_l, el); 
//}
//
//bool AnalysiG::core::basis_t::element(std::vector<std::vector<int>>* el){
//    return this -> _getalt(this -> vvv_i, this -> vv_i, el); 
//}
//
//bool AnalysiG::core::basis_t::element(std::vector<std::vector<bool>>* el){
//    return this -> _getalt(this -> vvv_b, this -> vv_b, el); 
//}
//
//
//
//bool AnalysiG::core::basis_t::element(std::vector<float>* el){
//    return this -> _getalt(this -> vv_f, this -> v_f, el); 
//}
//
//bool AnalysiG::core::basis_t::element(std::vector<double>* el){
//    return this -> _getalt(this -> vv_d, this -> v_d, el); 
//}
//
//bool AnalysiG::core::basis_t::element(std::vector<int>* el){
//    return this -> _getalt(this -> vv_i, this -> v_i, el); 
//}
//
//bool AnalysiG::core::basis_t::element(std::vector<bool>* el){
//    return this -> _getalt(this -> vv_b, this -> v_b, el); 
//}
//
//bool AnalysiG::core::basis_t::element(std::vector<long>* el){
//    return this -> _getalt(this -> vv_l, this -> v_l, el); 
//}
//
//bool AnalysiG::core::basis_t::element(std::vector<char>* el){
//    return this -> _getalt(this -> vv_c, this -> v_c, el); 
//}
//
//bool AnalysiG::core::basis_t::element(bool* el){
//    return this -> _getalt(this -> v_b, this -> b, el); 
//}
//
//bool AnalysiG::core::basis_t::element(double* el){
//    return this -> _getalt(this -> v_d, this -> d, el); 
//}
//
//bool AnalysiG::core::basis_t::element(float* el){
//    return this -> _getalt(this -> v_f, this -> f, el); 
//}
//
//bool AnalysiG::core::basis_t::element(int* el){
//    return this -> _getalt(this -> v_i, this -> i, el); 
//}
//
//bool AnalysiG::core::basis_t::element(long* el){
//    return this -> _getalt(this -> v_l, this -> l, el); 
//}
//
//bool AnalysiG::core::basis_t::element(unsigned long long* el){
//    return this -> _getalt(this -> v_ull, this -> ull, el); 
//}
//
//bool AnalysiG::core::basis_t::element(unsigned int* el){
//    return this -> _getalt(this -> v_ui, this -> ui, el); 
//}
//
//bool AnalysiG::core::basis_t::element(char* el){
//    return this -> _getalt(this -> v_c, this -> c, el); 
//}
//// ******************************************************************************************* //
//
//void variable_t::process(std::vector<std::vector<float>>* data, std::string* varname, TTree* tr){
//    this -> add_data(data, this -> vv_f, varname, tr);
//}
//
//void variable_t::process(std::vector<std::vector<double>>* data, std::string* varname, TTree* tr){
//    this -> add_data(data, this -> vv_d, varname, tr); 
//}
//
//void variable_t::process(std::vector<std::vector<long>>* data, std::string* varname, TTree* tr){
//    this -> add_data(data, this -> vv_l, varname, tr); 
//}
//
//void variable_t::process(std::vector<std::vector<int>>* data, std::string* varname, TTree* tr){
//    this -> add_data(data, this -> vv_i, varname, tr); 
//}
//
//void variable_t::process(std::vector<std::vector<bool>>* data, std::string* varname, TTree* tr){
//    this -> add_data(data, this -> vv_b, varname, tr); 
//}
//
//void variable_t::process(std::vector<float>* data, std::string* varname, TTree* tr){
//    this -> add_data(data, this -> v_f, varname, tr); 
//}
//
//void variable_t::process(std::vector<double>* data, std::string* varname, TTree* tr){
//    this -> add_data(data, this -> v_d, varname, tr); 
//}
//
//void variable_t::process(std::vector<long>* data, std::string* varname, TTree* tr){
//    this -> add_data(data, this -> v_l, varname, tr); 
//}
//
//void variable_t::process(std::vector<int>* data, std::string* varname, TTree* tr){
//    this -> add_data(data, this -> v_i, varname, tr); 
//}
//
//void variable_t::process(std::vector<bool>* data, std::string* varname, TTree* tr){
//    this -> add_data(data, this -> v_b, varname, tr); 
//}
//
//void variable_t::process(float* data, std::string* varname, TTree* tr){
//    this -> add_data(data, this -> f, varname, tr); 
//}
//
//void variable_t::process(double* data, std::string* varname, TTree* tr){
//    this -> add_data(data, this -> d, varname, tr); 
//}
//
//void variable_t::process(long* data, std::string* varname, TTree* tr){
//    this -> add_data(data, this -> l, varname, tr); 
//}
//
//void variable_t::process(int* data, std::string* varname, TTree* tr){
//    this -> add_data(data, this -> i, varname, tr); 
//}
//
//void variable_t::process(bool* data, std::string* varname, TTree* tr){
//    this -> add_data(data, this -> b, varname, tr); 
//}
//// ==================================================================================== //
//
