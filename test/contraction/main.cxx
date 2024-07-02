#include <iostream>
#include <stdio.h>
#include <vector>

template <typename g>
void scout_dim(g* data, int* mx_dim){return;}

template <typename G>
void scout_dim(std::vector<G>* vec, int* mx_dim){
    int dim_ = 0;
    for (int x(0); x < vec -> size(); ++x){
        scout_dim(&vec -> at(x), &dim_);
        if (!dim_){dim_ = vec -> size();}
    }
    if (dim_ < *mx_dim){return;}
    *mx_dim = dim_; 
};

// sink
template <typename g>
void nulls(g* d, int* mx_dim){*d = -1;}

template <typename g>
void nulls(std::vector<g>* d, int* mx_dim){
    for (int t(d -> size()); t < *mx_dim; ++t){
        d -> push_back({});
        nulls(&d -> at(t), mx_dim);
    }
}

template <typename g>
bool standard(g* data, int* mx_dim){ return true; }

template <typename g>
bool standard(std::vector<g>* vec, int* mx_dim){
    int l = vec -> size();
    if (!l){nulls(vec, mx_dim);}
    for (size_t x(0); x < l; ++x){
        if (!standard(&vec -> at(x), mx_dim)){continue;}
        nulls(vec, mx_dim);
        return false;
    };
    return false; 
}; 



template <typename G, typename g>
void as_primitive(G* data, std::vector<g>* lin, std::vector<int>* dims, int depth){
    lin -> push_back(*data);
}; 

template <typename G, typename g>
void as_primitive(std::vector<G>* data, std::vector<g>* linear, std::vector<int>* dims, int depth = 0){
    if (depth == dims -> size()){dims -> push_back(data -> size());}
    for (int x(0); x < data -> size(); ++x){
        as_primitive(&data -> at(x), linear, dims, depth+1);
    }
}; 

int main(){
    std::vector<std::vector<std::vector<int>>> truth_res = {
        {{-1, -1, -1, -1, -1, -1}, {10, 11, 12, -1, -1, -1}, {1, 2, 3, 4, 5, 6}}, 
        {{1, 2, 3,  4,  5,  6   }, {10, 11, 12, -1, -1, -1}, {1, 2, 3, 4, 5, 6}}, 
        {{1, 2, 3, -1, -1, -1   }, {10, 11, 12, -1, -1, -1}, {1, 2, 3, 4, 5, 6}}, 
        {{1, 2, 3,  4,  5,  6   }, {10, 11, 12, -1, -1, -1}, {1, 2, 3, 4, 5, 6}}
    }; 

    std::vector<int> data = {1, 2, 3, 4, 5, 6}; 
    std::vector<int> jagg = {10, 11, 12}; 
    std::vector<int> djad = {1, 2, 3}; 
    
    std::vector<std::vector<std::vector<int>>> exmpl = {
        {{}  , jagg, data},
        {data, jagg, data}, 
        {djad, jagg, data}, 
        {data, jagg, data}
    }; 

    int mx_dim = 0; 
    scout_dim(&exmpl, &mx_dim); 
    standard(&exmpl, &mx_dim); 

    std::vector<int> dims = {}; 
    std::vector<int> linear = {}; 
    as_primitive(&exmpl, &linear, &dims); 
    for (size_t x(0); x < dims.size(); ++x){std::cout << dims[x] << std::endl;}

    for (size_t x(0); x < truth_res.size(); ++x){
        for (size_t y(0); y < truth_res[x].size(); ++y){
            for (size_t z(0); z < truth_res[x][y].size(); ++z){
                if (exmpl[x][y][z] == linear[x+y+z]){continue;}
                if (exmpl[x][y][z] == truth_res[x][y][z]){continue;}
                std::cout << "(false)-> " << exmpl[x][y][z] << " " << truth_res[x][y][z] << std::endl;
            } 
        }
    }

    return 0; 
}; 
