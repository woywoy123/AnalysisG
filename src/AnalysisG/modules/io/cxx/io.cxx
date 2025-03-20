#include "io.h"

io::io(){this -> prefix = "io";}
io::~io(){
    this -> end();
    this -> root_end(); 
    std::map<std::string, TFile*>::iterator itr = this -> files_open.begin(); 
    for (; itr != this -> files_open.end(); ++itr){
        if (itr -> second -> IsOpen()){itr -> second -> Close();}
        delete itr -> second;
    }

    std::map<std::string, meta*>::iterator itm = this -> meta_data.begin(); 
    for (; itm != this -> meta_data.end(); ++itm){
        if (!itm -> second){continue;}
        delete itm -> second;
    }
    this -> meta_data.clear(); 
}

void io::import_settings(settings_t* params){
    this -> enable_pyami = params -> fetch_meta; 
    this -> metacache_path = params -> metacache_path; 
    this -> sow_name = params -> sow_name; 
    if (!this -> sow_name.size()){return;}
    this -> info("Checking for Sum of Weights under tree name: " + this -> sow_name); 
}
