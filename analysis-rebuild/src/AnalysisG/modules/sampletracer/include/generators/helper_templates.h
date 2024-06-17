#ifndef SAMPLETRACER_TEMPLATES_H
#define SAMPLETRACER_TEMPLATES_H

template<typename G>
void register_object(std::map<std::string, G*>* inpt, std::vector<std::map<std::string, G*>>* target, std::map<std::string, int>* hash_map){
    typename std::map<std::string, G*>::iterator itr = inpt -> begin(); 
    for (; itr != inpt -> end(); ++itr){
        std::string _hash = itr -> second -> hash;
        std::string _type = itr -> second -> name; 
        std::string _tree = itr -> second -> tree; 
        if (!hash_map -> count(_hash)){
            (*hash_map)[_hash] = (int)target -> size(); 
            target -> push_back({}); 
        }
        int index = (*hash_map)[_hash]; 
        if ((*target)[index].count(_tree + "-" + _type)){continue;}
        (*target)[index][_tree + "-" + _type] = itr -> second;
    }
}; 

template <typename G>
void find_object(std::vector<G*>* trg, std::vector<std::map<std::string, G*>>* src, std::string name, std::string tree){
    tools t = tools(); 
    for (size_t x(0); x < src -> size(); ++x){
        typename std::map<std::string, G*>::iterator itr = src -> at(x).begin();
        for (; itr != src -> at(x).end(); ++itr){
            if (!name.size() && !tree.size()){
                trg -> push_back(itr -> second); continue;
            }
            if (!t.has_string((std::string*)&itr -> first, name)){return;}
            if (!t.has_string((std::string*)&itr -> first, tree)){continue;}
            trg -> push_back(itr -> second); 
        }
    }
}; 

template <typename G>
void static process_data(std::vector<G*>* ev, bool* execute){
    bool comple = false; 
    while (*execute || !comple){
        for (int x(0); x < ev -> size(); ++x){
            if (!(*ev)[x]){continue;}
            (*ev)[x] -> CompileEvent();
            (*ev)[x] -> is_compiled = true;
        }
        comple = true;
        for (int x(0); x < ev -> size(); ++x){
            if (!(*ev)[x]){continue;}
            comple *= (*ev)[x] -> is_compiled; 
            if (!(*ev)[x] -> is_compiled){continue;}
            (*ev)[x] = nullptr; 
        }
    }
}; 

template <typename G>
void threaded_compiler(std::map<std::string, int>* hash_map, std::vector<std::map<std::string, G*>>* type_tree, int threads){
    typename std::map<std::string, G*>::iterator ite; 
    typename std::map<std::string, G*>* mps;
    std::map<std::string, int>::iterator itr; 
    
    int th_dx = 0; 
    int index = 0; 
    int threshold = 100; 
    bool execute = true; 
    std::map<int, std::vector<G*>*> event_compile = {}; 
    std::map<int, std::thread*> threads_running = {}; 
    for (int x(0); x < threads; ++x){
        event_compile[x] = new std::vector<G*>(threshold, nullptr);
        threads_running[x] = nullptr; 
    }

    for (itr = hash_map -> begin(); itr != hash_map -> end(); ++itr){
        mps = &type_tree -> at(itr -> second); 
        for (ite = mps -> begin(); ite != mps -> end(); ++ite){
            ite -> second -> CompileEvent(); 
            ite -> second -> is_compiled = true; 
            if (ite -> second -> is_compiled){continue;}

            std::vector<G*>* vec = event_compile[th_dx]; 
            G* ev_ = vec -> at(index); 
            if (!ev_){(*vec)[index] = ite -> second;}
            else {--ite;}
            index++; 

            if (!threads_running[th_dx]){
                threads_running[th_dx] = new std::thread(process_data<G>, vec, &execute);
                th_dx++; 
            }
            if (index >= threshold){index = 0; th_dx++;}
            if (th_dx >= threads){th_dx = 0; index = 0;}
        }
    }
    std::map<int, std::thread*>::iterator itx = threads_running.begin(); 
    for (; itx != threads_running.end(); ++itx){
        execute = false; 
        if (itx -> second){
            itx -> second -> join();
            delete itx -> second; 
        }
        event_compile[itx -> first] -> clear(); 
        delete event_compile[itx -> first]; 
    }
}; 

#endif
