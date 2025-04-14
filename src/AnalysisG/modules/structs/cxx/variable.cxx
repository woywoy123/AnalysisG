#include <structs/element.h>

void write_t::write(){
    this -> tree -> Fill(); 
    std::map<std::string, variable_t*>::iterator itx = this -> data -> begin(); 
    for (; itx != this -> data -> end(); ++itx){itx -> second -> flush_buffer();}
}

void write_t::create(std::string tr_name, std::string path){
    if (!this -> file){this -> file = new TFile(path.c_str(), "RECREATE");}
    if (this -> mtx){
        this -> tree = new TTree("MetaData", "meta"); 
        this -> tree -> Branch("MetaData", this -> mtx); 
        this -> tree -> Fill();
        this -> tree -> Write("", TObject::kOverwrite);  
        delete this -> tree; 
        this -> tree = nullptr; 
        this -> mtx = nullptr; 
    }
    if (this -> data){return;}
    this -> tree = new TTree(tr_name.c_str(), "data"); 
    this -> data = new std::map<std::string, variable_t*>(); 
}

void write_t::close(){
    if (this -> tree){
        this -> tree -> ResetBranchAddresses(); 
        this -> tree -> Write("", TObject::kOverwrite); 
        delete this -> tree; 
        this -> tree = nullptr; 
    }
    if (this -> file){
        this -> file -> Close();
        this -> file -> Delete(); 
        delete this -> file; 
        this -> file = nullptr;
    }
    if (!this -> data){return;}
    std::map<std::string, variable_t*>::iterator itx = this -> data -> begin(); 
    for (; itx != this -> data -> end(); ++itx){
        itx -> second -> clear = true; 
        delete itx -> second;
    }
    this -> data -> clear(); 
    delete this -> data; 
}

variable_t* write_t::process(std::string* name){
    if (this -> data -> count(*name)){return (*this -> data)[*name];}
    variable_t* t = new variable_t(); 
    t -> tt = this -> tree; 
    (*this -> data)[*name] = t; 
    return t; 
}



writer::writer(){}
writer::~writer(){
    std::map<std::string, write_t*>::iterator it = this -> handle.begin();
    for (; it != this -> handle.end(); ++it){
        it -> second -> file = nullptr;
        it -> second -> close(); 
        delete it -> second; it -> second = nullptr; 
    }
    this -> handle.clear(); 
    this -> head -> close();
    delete this -> head;
}

void writer::create(std::string* out){
    if (this -> head){return;}
    this -> head = new write_t();
    this -> head -> file = new TFile(out -> c_str(), "UPDATE");     
}

variable_t* writer::process(std::string* tree, std::string* name){
    if (this -> handle.count(*tree)){return this -> handle[*tree] -> process(name);}
    write_t* wr = new write_t(); 
    wr -> file = this -> head -> file; 
    wr -> create(*tree, ""); 
    this -> handle[*tree] = wr; 
    return this -> process(tree, name);
}

void writer::write(std::string* tree){
    if (!this -> handle.count(*tree)){return;}
    this -> handle[*tree] -> write(); 
}











