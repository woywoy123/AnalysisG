#ifndef CODE_H
#define CODE_H
#include <iostream>
#include <vector>
#include <string>
#include <map>

namespace Tools
{
    class CyCode
    {
        public:
            CyCode();
            ~CyCode();
            void Hash();
            bool operator==(CyCode* code);

            std::vector<std::string> input_params = {};
            std::vector<std::string> co_vars = {};

            std::map<std::string, std::string> param_space = {};

            std::string function_name = "";
            std::string class_name = "";
            std::string hash = "";

            std::string source_code = "";
            std::string object_code = "";

            bool is_class = false;
            bool is_function = false;

            bool is_callable = false;
            bool is_initialized = false;

            bool has_param_variable = false;
    };
}
#endif
