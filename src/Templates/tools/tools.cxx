#include "../tools/tools.h"

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


