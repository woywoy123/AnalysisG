template <typename g, typename k>
g* AnalysisG::tooling::as(k* ip){return (g*)ip;}

template <typename g>
std::string AnalysisG::tooling::to_string(g val){
    std::stringstream ss; 
    ss << val; 
    return ss.str(); 
} 

template <typename G>
std::vector<std::vector<G>> AnalysisG::tooling::discretize(std::vector<G>* v, int N){
    size_t n = v -> size(); 
    typename std::vector<std::vector<G>> out; 
    out.reserve(int(v -> size()/N)); 
    for (size_t ib = 0; ib < n; ib += N){
        size_t end = ib + N; 
        if (end > n){ end = n; }
        out.push_back(std::vector<G>(v -> begin() + ib, v -> begin() + end)); 
    }
    return out; 
}

template <typename g>
g AnalysisG::tooling::max(std::vector<g>* inpt){
    g ix = inpt -> at(0); 
    for (size_t t(1); t < inpt -> size(); ++t){
        if (inpt -> at(t) <= ix){continue;}
        ix = inpt -> at(t); 
    }
    return ix; 
}

template <typename g>
g AnalysisG::tooling::min(std::vector<g>* inpt){
    g ix = inpt -> at(0); 
    for (size_t t(1); t < inpt -> size(); ++t){
        if (inpt -> at(t) >= ix){continue;}
        ix = inpt -> at(t); 
    }
    return ix; 
}

template <typename g>
std::vector<g*> AnalysisG::tooling::put(std::vector<g*>* src, std::vector<unsigned long>* trg){
    typename std::vector<g*> out(src -> size(), nullptr); 
    for (size_t x(0); x < trg -> size(); ++x){out[x] = (*src)[(*trg)[x]];}
    return out; 
}

template <typename g>
void AnalysisG::tooling::put(std::vector<g*>* out, std::vector<g*>* src, std::vector<unsigned long>* trg){
    out -> clear(); 
    out -> reserve(trg -> size());
    for (size_t x(0); x < trg -> size(); ++x){
        g* v = (*src)[(*trg)[x]];  
        out -> push_back(v);
        v -> in_use = 1; 
    }
}

template <typename g>
void AnalysisG::tooling::count(g* inp, long* ix){*ix += 1;}

template <typename g>
void AnalysisG::tooling::count(std::vector<g>* inp, long* ix){
    for (size_t x(0); x < inp -> size(); ++x){AnalysisG::tooling::count(&inp -> at(x), ix);}
}

template <typename g>
void AnalysisG::tooling::contract(std::vector<g>* out, g* p2){out -> push_back(*p2);}

template <typename g>
void AnalysisG::tooling::contract(std::vector<g>* out, std::vector<g>* p2){
    for (size_t i(0); i < p2 -> size(); ++i){AnalysisG::tooling::contract(out, &p2 -> at(i));}
}

template <typename g>
void AnalysisG::tooling::contract(std::vector<g>* out, std::vector<std::vector<g>>* p2){
    long ix = 0;
    AnalysisG::tooling::count(p2, &ix);
    out -> reserve(ix); 
    for (size_t i(0); i < p2 -> size(); ++i){AnalysisG::tooling::contract(out, &p2 -> at(i));}
}

// ----- needed for cython ----- //
template <typename g>
void AnalysisG::tooling::release_vector(std::vector<g>* ipt){ ipt -> shrink_to_fit(); }

