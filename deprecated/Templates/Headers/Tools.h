#include <iostream>
#include <sstream> 
#include <vector>
#include <string>

#ifndef TOOLS_H
#define TOOLS_H

namespace Tools
{
    std::string Hashing(std::string inpt); 
    std::string ToString(double inpt); 
    std::vector<std::string> Split(const std::string &s, std::string delim);

    std::vector<std::vector<std::string>> Chunk(const std::vector<std::string>& v, int N); 
}
#endif 
