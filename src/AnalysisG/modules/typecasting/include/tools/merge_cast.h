#ifndef TYPECASTING_MERGE_CAST_H
#define TYPECASTING_MERGE_CAST_H

#include <string>
#include <vector>
#include <map>

template <typename G>
void merge_data(std::vector<G>* out, std::vector<G>* p2){
    out -> insert(out -> end(), p2 -> begin(), p2 -> end()); 
}

template <typename G>
void merge_data(std::map<std::string, G>* out, std::map<std::string, G>* p2){
    typename std::map<std::string, G>::iterator itr = p2 -> begin(); 
    for (; itr != p2 -> end(); ++itr){merge_data(&(*out)[itr -> first], &itr -> second);} 
}

#endif