#include "../tools/tools.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>

#ifndef CODE_H
#define CODE_H

struct ExportCode
{
    std::vector<std::string> input_params; 
    std::vector<std::string> co_vars; 

    std::map<std::string, std::string> param_space; 

    std::string function_name;
    std::string class_name;
    std::string hash; 
    std::string source_code; 
    std::string object_code; 
    
    bool is_class;
    bool is_function; 
    bool is_callable; 
    bool is_initialized; 
    bool has_param_variable; 

}; 

namespace Code
{
    class CyCode
    {
        public:
            CyCode();
            ~CyCode();
            void Hash();
            bool operator==(CyCode* code);
            ExportCode MakeMapping(); 
            void ImportCode(ExportCode code);

            std::vector<std::string> input_params = {};
            std::vector<std::string> co_vars = {};

            std::map<std::string, std::string> param_space = {};

            std::string function_name = "";
            std::string class_name = "";
            std::string hash = "";

            std::string source_code = "";
            std::string object_code = "";
            std::map<std::string, ExportCode> dependency = {}; 

            bool is_class = false;
            bool is_function = false;

            bool is_callable = false;
            bool is_initialized = false;

            bool has_param_variable = false;
    };
}

#endif
