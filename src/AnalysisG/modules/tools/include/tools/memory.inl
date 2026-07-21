template <typename g>
bool AnalysisG::tooling::pflush(g** p){
    if (!(*p)){return true;}
    delete *p; 
    (*p) = nullptr; 
    return false; 
}

template <typename g>
void AnalysisG::tooling::vflush(std::vector<g*>* data, bool only_null){
    if (!data){return;}
    for (size_t x(0); x < data -> size(); ++x){AnalysisG::tooling::pflush(&(*data)[x]);}
    if (only_null){return;}
    data -> clear(); 
    data -> shrink_to_fit(); 
}

template <typename k, typename g>
void AnalysisG::tooling::vflush(std::vector< std::map<k, g>* >* data){
    if (!data){return;}
    for (size_t x(0); x < data -> size(); ++x){AnalysisG::tooling::pflush(&(*data)[x]);}
    data -> clear(); 
    data -> shrink_to_fit(); 
}

template <typename k, typename g>
void AnalysisG::tooling::mflush(std::map<k, std::vector<g*>*>* data){
    if (!data){return;}
    typename std::map<k, std::vector<g*>*>::iterator ix; 
    for (ix = data -> begin(); ix != data -> end(); ++ix){
        AnalysisG::tooling::vflush(ix -> second); 
        AnalysisG::tooling::pflush(&ix -> second); 
    }
    data -> clear(); 
}

template <typename k, typename g>
void AnalysisG::tooling::mflush(std::map<k, g*>* data, bool only_null){
    if (!data){return;}
    typename std::map<k, g*>::iterator ix; 
    for (ix = data -> begin(); ix != data -> end(); ++ix){
        if (!ix -> second){continue;}
        AnalysisG::tooling::pflush(&ix -> second); 
    }
    if (only_null){return;}
    data -> clear(); 
}

template <typename k, typename g>
void AnalysisG::tooling::mflush(std::map<k, std::vector<g>*>* data){
    if (!data){return;}
    typename std::map< k, std::vector<g>* >::iterator ix; 
    for (ix = data -> begin(); ix != data -> end(); ++ix){
        if (!ix -> second){continue;}
        AnalysisG::tooling::pflush(&ix -> second); 
    }
    data -> clear(); 
}


