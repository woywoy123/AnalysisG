#include "../tools/tools.h"
#include <vector>

std::string Hashing(std::string input)
{
    std::hash<std::string> hasher; 
    std::stringstream ss; 
    ss << "0x" << std::hex << hasher(input); 
    std::string out = ss.str(); 
    int diff = out.size() - 18; 
    if (!diff) { return out; }
    out += std::string(std::abs(diff), '0'); 
    return out; 
}

std::string ToString(double inpt)
{
    std::stringstream ss; 
    ss << inpt; 
    return ss.str(); 
}

std::vector<std::string> split(std::string inpt, std::string search)
{
    size_t pos = 0;
    size_t s_dim = search.length(); 
    size_t index = 0; 
    std::string token; 
    std::vector<std::string> out = {}; 
    while ((pos = inpt.find(search)) != std::string::npos)
    {
        out.push_back(inpt.substr(0, pos));
        inpt.erase(0, pos + s_dim); 
        ++index; 
    }
    out.push_back(inpt); 
    return out; 
}

std::string join(std::vector<std::string>* inpt, int index_s, int index_e, std::string delim)
{
    std::string out = ""; 
    if (index_e < 0){ index_e = inpt -> size(); }
    for (int i(index_s); i < index_e-1; ++i){ out += inpt -> at(i) + delim; }
    out += inpt -> at(index_e-1); 
    return out; 
}

int count(std::string inpt, std::string search)
{
    int index = 0; 
    int s_size = search.length(); 
    if (!s_size){return 0;}

    std::string::size_type i = inpt.find(search); 
    while ( i != std::string::npos)
    {
        ++index; 
        i = inpt.find(search, i + s_size); 
    }
    return index; 
}

std::vector<std::vector<std::string>> Quantize(const std::vector<std::string>& v, int N)
{
    int n = v.size(); 
    int size_max = n/N + (n % N != 0); 
    std::vector<std::vector<std::string>> out; 
    for (int ib = 0; ib < n; ib += size_max)
    {
        int end = ib + size_max; 
        if (end > n){ end = n; }
        out.push_back(std::vector<std::string>(v.begin() + ib, v.begin() + end)); 
    }
    return out; 
}

