#include <iostream>
#include <string>
#include <vector>
#include <cstdint>

#ifndef TOOLS_TOOLS_H
#define TOOLS_TOOLS_H

class tools
{
    public:
        // tools.cxx
        tools(); 
        ~tools(); 

        // io.cxx
        void create_path(std::string path); 
        void delete_path(std::string path); 
        bool is_file(std::string path); 

        // strings.cxx
        void replace(std::string* in, std::string repl_str, std::string repl_with); 
        std::vector<std::string> split(std::string in, std::string delim);
        std::vector<std::string> split(std::string in, int n);
        std::string hash(std::string input, int len = 18);
        std::string upper(std::string*); 
        std::string lower(std::string*); 
        std::string capitalize(std::string*);
        std::string urlencode(const std::string& value);
        bool has_string(std::string* inpt, std::string trg); 

        // conversion.cxx
        void to_uint8(std::string* in, uint8_t* out);
        std::vector<uint8_t> to_uint8(std::string in); 
        std::vector<uint8_t> to_uint8(std::string* in);
        void to_hex(uint8_t* in, unsigned int len, std::string* out); 
        std::string to_hex(std::string* in); 
        std::string hex_to_string(std::string* in);

        std::string to_string(double qnt, int prec = -1); 
        std::string remove_leading(const std::string& input, const std::string& leader = "0");
        std::string remove_trailing(std::string* inpt); 

        // Template functions
        template <typename G>
        std::vector<std::vector<G>> discretize(std::vector<G>* v, int N){
            int n = v -> size(); 
            typename std::vector<std::vector<G>> out; 
            for (int ib = 0; ib < n; ib += N){
                int end = ib + N; 
                if (end > n){ end = n; }
                out.push_back(std::vector<G>(v -> begin() + ib, v -> begin() + end)); 
            }
            return out; 
        };
}; 


#endif
