template <typename g>
void AnalysisG::tooling::scout_dim(const g*, int*){return;}

template <typename G>
void AnalysisG::tooling::scout_dim(const std::vector<G>* vec, int* mx_dim){
    int dim_ = 0;
    for (size_t x(0); x < vec -> size(); ++x){
        AnalysisG::tooling::scout_dim(&vec -> at(x), &dim_);
        if (!dim_){dim_ = vec -> size();}
    }
    if (dim_ < *mx_dim){return;}
    *mx_dim = dim_; 
}

template <typename g>
void AnalysisG::tooling::nulls(g* d, int*){*d = -1;}

template <typename g>
void AnalysisG::tooling::nulls(const std::vector<g>* d, int* mx_dim){
    for (size_t t(d -> size()); t < *mx_dim; ++t){
        d -> push_back({});
        AnalysisG::tooling::nulls(&d -> at(t), mx_dim);
    }
} 

template <typename g>
bool AnalysisG::tooling::standard(const g*, int*){ return true; }

template <typename g>
bool AnalysisG::tooling::standard(const std::vector<g>* vec, int* mx_dim){
    size_t l = vec -> size();
    if (!l){AnalysisG::tooling::nulls(vec, mx_dim);}
    for (size_t x(0); x < l; ++x){
        if (!AnalysisG::tooling::standard(&vec -> at(x), mx_dim)){continue;}
        AnalysisG::tooling::nulls(vec, mx_dim);
        return false;
    };
    return false; 
}

template <typename G, typename g>
void AnalysisG::tooling::as_primitive(G* data, std::vector<g>* lin, std::vector<signed long>*, unsigned int){
    lin -> push_back(*data);
} 

template <typename G, typename g>
void AnalysisG::tooling::as_primitive(std::vector<G>* data, std::vector<g>* linear, std::vector<signed long>* dims, unsigned int depth){
    if (depth == dims -> size()){dims -> push_back(data -> size());}
    for (size_t x(0); x < data -> size(); ++x){
        AnalysisG::tooling::as_primitive(&(*data)[x], linear, dims, depth+1);
    }
} 

