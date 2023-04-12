#include "../Headers/Tools.h"

std::string Tools::Hashing(std::string inpt)
{
    std::hash<std::string> hasher; 
    std::stringstream ss; 
    ss << "0x" << std::hex << hasher(inpt); 
    return ss.str(); 
}

std::string Tools::ToString(double inpt)
{
    std::stringstream ss; 
    ss << inpt;
    return ss.str(); 
}
