#ifndef TYPECASTING_VECTOR_CAST_H
#define TYPECASTING_VECTOR_CAST_H

#include <torch/torch.h>
#include <vector>
#include <TTree.h>

std::vector<signed long> tensor_size(torch::Tensor* inpt){
    c10::IntArrayRef dims = inpt -> sizes();
    std::vector<signed long> out; 
    for (size_t x(0); x < dims.size(); ++x){out.push_back(dims[x]);}
    return out;  
}

template <typename G>
std::vector<std::vector<G>> chunking(std::vector<G>* v, int N){
    int n = v -> size(); 
    typename std::vector<std::vector<G>> out; 
    for (int ib = 0; ib < n; ib += N){
        int end = ib + N; 
        if (end > n){ end = n; }
        out.push_back(std::vector<G>(v -> begin() + ib, v -> begin() + end)); 
    }
    return out; 
}

template <typename g>
void tensor_vector(std::vector<g>* trgt, std::vector<g>* chnks, std::vector<signed long>* dims, int next_dim = 0){
    trgt -> insert(trgt -> end(), chnks -> begin(), chnks -> end());  
}


template <typename G, typename g>
void tensor_vector(std::vector<G>* trgt, std::vector<g>* chnks, std::vector<signed long>* dims, int next_dim = 0){
    std::vector<std::vector<g>> chnk_n = chunking(chnks, (*dims)[next_dim]);
    for (size_t x(0); x < chnk_n.size(); ++x){
        G tmp = {}; 
        tensor_vector(&tmp, &chnk_n[x], dims, next_dim-1); 
        trgt -> push_back(tmp); 
    }
}

template <typename G, typename g>
void tensor_to_vector(torch::Tensor* data, std::vector<G>* out, std::vector<signed long>* dims, g prim){
    torch::Tensor tens = data -> view({-1}).to(torch::kCPU); 
    typename std::vector<g> linear(tens.data_ptr<g>(), tens.data_ptr<g>() + tens.numel()); 
    tensor_vector(out, &linear, dims, dims -> size()-1); 
}

void add_to_dict(std::vector<std::vector<float>>* dummy){gInterpreter -> GenerateDictionary("vector<vector<float>>", "vector");}
void add_to_dict(std::vector<std::vector<double>>* dummy){gInterpreter -> GenerateDictionary("vector<vector<double>>", "vector");}
void add_to_dict(std::vector<std::vector<long>>* dummy){gInterpreter -> GenerateDictionary("vector<vector<long>>", "vector");}
void add_to_dict(std::vector<std::vector<int>>* dummy){gInterpreter -> GenerateDictionary("vector<vector<int>>", "vector");}

void add_to_dict(std::vector<float>* dummy){gInterpreter -> GenerateDictionary("vector<float>", "vector");}
void add_to_dict(std::vector<double>* dummy){gInterpreter -> GenerateDictionary("vector<double>", "vector");}
void add_to_dict(std::vector<long>* dummy){gInterpreter -> GenerateDictionary("vector<long>", "vector");}
void add_to_dict(std::vector<int>* dummy){gInterpreter -> GenerateDictionary("vector<int>", "vector");}
void add_to_dict(std::vector<bool>* dummy){gInterpreter -> GenerateDictionary("vector<bool>", "vector");}

struct variable_t {
    public:
        void process(torch::Tensor* data, std::string varname, TTree* tr){
            this -> tt = tr;
            std::vector<signed long> s = tensor_size(data); 
                
            // type and dim switch for the tensors
            if (s.size() == 2 && data -> dtype() == torch::kDouble){
                return this -> add_data(&this -> vvd, data, &s, &varname, double(0));
            }

            if (s.size() == 1 && data -> dtype() == torch::kDouble){
                return this -> add_data(&this -> vd, data, &s, &varname, double(0));
            }

            if (s.size() == 2 && data -> dtype() == torch::kFloat32){
                return this -> add_data(&this -> vvf, data, &s, &varname, float(0));
            }

            if (s.size() == 1 && data -> dtype() == torch::kFloat32){
                return this -> add_data(&this -> vf, data, &s, &varname, float(0));
            }

            if (s.size() == 2 && data -> dtype() == torch::kLong){
                return this -> add_data(&this -> vvl, data, &s, &varname, long(0)); 
            }

            if (s.size() == 1 && data -> dtype() == torch::kLong){
                return this -> add_data(&this -> vl, data, &s, &varname, long(0)); 
            }

            if (s.size() == 2 && data -> dtype() == torch::kInt){
                return this -> add_data(&this -> vvi, data, &s, &varname, int(0)); 
            }

            if (s.size() == 1 && data -> dtype() == torch::kInt){
                return this -> add_data(&this -> vi, data, &s, &varname, int(0)); 
            }

            if (s.size() == 1 && data -> dtype() == torch::kBool){
                return this -> add_data(&this -> vb, data, &s, &varname, bool(0)); 
            }

            std::cout << "DIM: " << s.size() << std::endl;
            std::cout << "Tensor Type: " << data -> dtype() << std::endl; 
            std::cout << *data << std::endl; 
            std::cout << "UNDEFINED DATA TYPE! SEE vector_cast.h" << std::endl;
            abort(); 
        }

        void purge(){
            if (this -> vvf){delete this -> vvf; return; }
            if (this -> vvd){delete this -> vvd; return; }
            if (this -> vvl){delete this -> vvl; return; }
            if (this -> vvi){delete this -> vvi; return; }

            if (this -> vf){delete this -> vf; return; }
            if (this -> vd){delete this -> vd; return; }
            if (this -> vl){delete this -> vl; return; }
            if (this -> vi){delete this -> vi; return; }
            if (this -> vb){delete this -> vb; return; }
        }

        void flush(){
            if (this -> vvf){this -> vvf -> clear(); return; }
            if (this -> vvd){this -> vvd -> clear(); return; }
            if (this -> vvl){this -> vvl -> clear(); return; }
            if (this -> vvi){this -> vvi -> clear(); return; }

            if (this -> vf){this -> vf -> clear(); return; }
            if (this -> vd){this -> vd -> clear(); return; }
            if (this -> vl){this -> vl -> clear(); return; }
            if (this -> vi){this -> vi -> clear(); return; }
            if (this -> vb){this -> vb -> clear(); return; }
        }

    private: 
        std::vector<std::vector<float>>*  vvf = nullptr; 
        std::vector<std::vector<double>>* vvd = nullptr; 
        std::vector<std::vector<long>>*   vvl = nullptr; 
        std::vector<std::vector<int>>*    vvi = nullptr; 

        std::vector<float>*                vf = nullptr; 
        std::vector<double>*               vd = nullptr; 
        std::vector<long>*                 vl = nullptr; 
        std::vector<int>*                  vi = nullptr; 
        std::vector<bool>*                 vb = nullptr; 

        TBranch*                           tb = nullptr; 
        TTree*                             tt = nullptr; 

        template <typename g, typename p>
        void add_data(std::vector<g>** type, torch::Tensor* data, std::vector<signed long>* s, std::string* varname, p prim){
            if (!(*type)){*type = new std::vector<g>();}
            tensor_to_vector(data, *type, s, prim); 
            if (this -> tb){return;}
            if (!this -> tt){return add_to_dict(*type);}
            this -> tb = this -> tt -> Branch(varname -> c_str(), *type); 
        }
};

#endif
