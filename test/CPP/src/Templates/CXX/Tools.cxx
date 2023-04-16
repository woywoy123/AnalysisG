#include "../Headers/Tools.h"

std::string Tools::Hashing(std::string inpt)
{
    std::hash<std::string> hasher; 
    std::stringstream ss; 
    ss << "0x" << std::hex << hasher(inpt); 
    std::string out = ss.str(); 
    int dif = out.size() - 18; 
    if (!dif){ return out; }
    out += std::string(std::abs(dif), '0'); 
    return out; 
}

std::string Tools::ToString(double inpt)
{
    std::stringstream ss; 
    ss << inpt;
    return ss.str(); 
}

std::string Tools::ToString(signed int inpt)
{
    std::stringstream ss; 
    ss << inpt;
    return ss.str(); 
}

std::vector<std::string> Tools::Split(const std::string &s, char delim)
{
    std::vector<std::string> r; 
    std::stringstream ss (s); 
    std::string item; 
    while (getline(ss, item, delim)){ r.push_back(item); }
    return r; 
}
