#include <iostream>
#include <sstream>

#ifndef TOOLS_H
#define TOOLS_H
#include <vector>

std::string Hashing(std::string input); 
std::string ToString(double input);
std::vector<std::string> split(std::string inpt, std::string search); 
std::string join(std::vector<std::string>* inpt, int s, int e, std::string delim); 
int count(std::string inpt, std::string search); 
std::vector<std::vector<std::string>> Quantize(const std::vector<std::string>& v, int N); 

#endif
