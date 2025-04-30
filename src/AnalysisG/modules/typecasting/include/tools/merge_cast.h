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
void merge_data(G* out, G* p2){(*out) = *p2;}

template <typename g, typename G>
void merge_data(std::map<g, G>* out, std::map<g, G>* p2){
    typename std::map<g, G>::iterator itr = p2 -> begin(); 
    for (; itr != p2 -> end(); ++itr){merge_data(&(*out)[itr -> first], &itr -> second);} 
}


template <typename G>
void sum_data(G* out, G* p2){(*out) += (*p2);}

template <typename G>
void sum_data(std::vector<G>* out, std::vector<G>* p2){
    out -> insert(out -> end(), p2 -> begin(), p2 -> end()); 
}

template <typename g, typename G>
void sum_data(std::map<g, G>* out, std::map<g, G>* p2){
    typename std::map<g, G>::iterator itr = p2 -> begin(); 
    for (; itr != p2 -> end(); ++itr){sum_data(&(*out)[itr -> first], &itr -> second);} 
}


template <typename g>
void reserve_count(g* inp, long* ix){*ix += 1;}

template <typename g>
void reserve_count(std::vector<g>* inp, long* ix){
    for (size_t x(0); x < inp -> size(); ++x){reserve_count(&inp -> at(x), ix);}
}

template <typename g>
void contract_data(std::vector<g>* out, g* p2){out -> push_back(*p2);}

template <typename g>
void contract_data(std::vector<g>* out, std::vector<g>* p2){
    for (size_t i(0); i < p2 -> size(); ++i){contract_data(out, &p2 -> at(i));}
}


template <typename g>
void contract_data(std::vector<g>* out, std::vector<std::vector<g>>* p2){
    long ix = 0;
    reserve_count(p2, &ix);
    out -> reserve(ix); 
    for (size_t i(0); i < p2 -> size(); ++i){contract_data(out, &p2 -> at(i));}
}

// ----- needed for cython ----- //
template <typename g>
void release_vector(std::vector<g>* ipt){ ipt -> shrink_to_fit(); }



#endif
