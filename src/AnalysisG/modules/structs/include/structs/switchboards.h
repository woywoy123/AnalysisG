#ifndef STRUCTS_SWITCH_H
#define STRUCTS_SWITCH_H

#include <string>
#include <map>

enum class mode_enum; 
enum class graph_enum; 


mode_enum model_mode(std::string* val); 
std::map<mode_enum, std::string> model_mode(std::map<std::string, std::string>* val); 
std::string enums_to_string(graph_enum gr); 


#endif
