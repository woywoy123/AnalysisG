#include <iostream>
#include <sstream> 
#include <vector>

#ifndef TOOLS_H
#define TOOLS_H

namespace Tools
{
    std::string Hashing(std::string inpt); 
    std::string ToString(double inpt); 
    std::string ToString(signed int inpt); 
    std::vector<std::string> Split(const std::string &s, char delim); 
}
#endif 
