#include "../Headers/Tools.h"

std::string Tools::Hashing(std::string str)
{
    unsigned long hash = std::hash<std::string>{}(str);     
    std::stringstream ss; 
    ss << std::hex << hash; 
    return ss.str(); 
}
