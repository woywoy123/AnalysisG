#include <element.h>

void element_t::set_meta(){
    std::map<std::string, data_t*>::iterator itr = this -> handle.begin();
    bool sk = itr -> second -> file_index >= (int)itr -> second -> files_i -> size(); 
    if (sk){return;}
    this -> event_index = itr -> second -> index; 
    this -> filename = itr -> second -> files_s -> at(itr -> second -> file_index);
}

bool element_t::next(){
    bool stop = true; 
    std::map<std::string, data_t*>::iterator itr = this -> handle.begin(); 
    for (; itr != this -> handle.end(); ++itr){stop *= itr -> second -> next();}
    return stop; 
}

bool element_t::boundary(){
    long idx = -1; 
    std::map<std::string, data_t*>::iterator itr = this -> handle.begin(); 
    for (; itr != this -> handle.end(); ++itr){idx = (*itr -> second -> files_i)[itr -> second -> file_index];}
    return idx > 0; 
}

void data_t::flush(){
    this -> flush_buffer();
    for (size_t x(0); x < this -> files_t -> size(); ++x){
        if (!(*this -> files_t)[x]){continue;}
        (*this -> files_t)[x] -> Close(); 
        (*this -> files_t)[x] -> Delete(); 
        delete (*this -> files_t)[x]; 
        (*this -> files_t)[x] = nullptr; 
    }
    this -> leaf = nullptr; 
    this -> branch = nullptr; 
    this -> tree = nullptr; 
    if (this -> files_s){delete this -> files_s; this -> files_s = nullptr;}
    if (this -> files_i){delete this -> files_i; this -> files_i = nullptr;}
    if (this -> files_t){delete this -> files_t; this -> files_t = nullptr;}
}

void data_t::initialize(){
    TFile* c = (*this -> files_t)[this -> file_index]; 
    c = (c -> Open(c -> GetTitle())); 
    this -> tree        = (TTree*)c -> Get(this -> tree_name.c_str()); 
    this -> tree -> SetCacheSize(10000000U); 
    this -> tree -> AddBranchToCache("*", true);
    this -> leaf        = this -> tree -> FindLeaf(this -> leaf_name.c_str());
    this -> branch      = this -> leaf -> GetBranch();  
    
    this -> tree_name   = this -> tree -> GetName();
    this -> leaf_name   = this -> leaf -> GetName();
    this -> branch_name = this -> branch -> GetName(); 

    this -> string_type(); 
    this -> flush_buffer(); 
    this -> fetch_buffer(); 
    this -> index = 0; 
    c -> Close(); 
    c -> Delete(); 
    delete c; 
    (*this -> files_t)[this -> file_index] = nullptr; 
} 

bool data_t::next(){
    if (this -> file_index >= (int)this -> files_i -> size()){return true;} 
    long idx = (*this -> files_i)[this -> file_index];
    this -> fname = &(*this -> files_s)[this -> file_index];
    if (this -> index+1 < idx){this -> index++; return false;}

    this -> file_index++; 
    if (this -> file_index >= (int)this -> files_i -> size()){return true;}
    this -> initialize();
    return false; 
}

void write_t::write(){
    this -> tree -> Fill(); 
    std::map<std::string, variable_t>::iterator itx = this -> data.begin(); 
    for (; itx != this -> data.end(); ++itx){itx -> second.flush();}
}

void write_t::create(std::string tr_name, std::string path){
    if (this -> file){return;}
    this -> file = new TFile(path.c_str(), "RECREATE"); 
    this -> tree = new TTree(tr_name.c_str(), "data"); 
}

void write_t::close(){
    this -> tree -> ResetBranchAddresses(); 
    this -> tree -> Write("", TObject::kOverwrite); 
    this -> file -> Close();
    this -> file -> Delete(); 
}


