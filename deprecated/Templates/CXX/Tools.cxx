#include <iostream>
#include <sstream> 
#include <vector>
#include <string>
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

std::vector<std::string> Tools::Split(const std::string &s, std::string delim)
{
    std::vector<std::string> r; 
    std::string spl = s;
    size_t pos = 0; 
    while ((pos = spl.find(delim)) != std::string::npos)
    {
        r.push_back(spl.substr(0, pos)); 
        spl.erase(0, pos + delim.length());  
    }
    r.push_back(spl);
    return r; 
}

std::vector<std::vector<std::string>> Tools::Chunk(const std::vector<std::string>& v, int N)
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


