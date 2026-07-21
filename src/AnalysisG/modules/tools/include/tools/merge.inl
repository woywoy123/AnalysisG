template <typename G>
void AnalysisG::tooling::merge(G* out, G* p2){(*out) = *p2;}

template <typename g, typename G>
void AnalysisG::tooling::merge(std::map<g, G>* out, std::map<g, G>* p2){
    typename std::map<g, G>::iterator itr = p2 -> begin(); 
    for (; itr != p2 -> end(); ++itr){AnalysisG::tooling::merge(&(*out)[itr -> first], &itr -> second);} 
}

template <typename G>
void AnalysisG::tooling::merge(std::vector<G>* out, std::vector<G>* p2){
    out -> insert(out -> end(), p2 -> begin(), p2 -> end()); 
}

template <typename g>
void AnalysisG::tooling::unique_key(std::vector<g>* inx, std::vector<g>* oth){
    typename std::map<g, bool> ch;
    for (size_t x(0); x < oth -> size(); ++x){ch[(*oth)[x]] = true;}
    for (size_t x(0); x < inx -> size(); ++x){
        g kx = (*inx)[x];
        if (ch[kx]){continue;}
        oth -> push_back(kx);
        ch[kx] = true;
    }
}


template <typename g>
g AnalysisG::tooling::sum(std::vector<g>* inpt){
    g ix = 0; 
    for (size_t t(0); t < inpt -> size(); ++t){ix += (*inpt)[t];}
    return ix; 
}

template <typename G>
void AnalysisG::tooling::sum(G* out, G* p2){(*out) += (*p2);}

template <typename G>
void AnalysisG::tooling::sum(std::vector<G>* out, std::vector<G>* p2){
    out -> insert(out -> end(), p2 -> begin(), p2 -> end()); 
}

template <typename g, typename G>
void AnalysisG::tooling::sum(std::map<g, G>* out, std::map<g, G>* p2){
    typename std::map<g, G>::iterator itr = p2 -> begin(); 
    for (; itr != p2 -> end(); ++itr){AnalysisG::tooling::sum(&(*out)[itr -> first], &itr -> second);} 
}


