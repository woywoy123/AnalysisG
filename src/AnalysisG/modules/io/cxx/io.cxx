#include <io/io.h>
#include <io/cfg.h>
#include <TSystem.h>
#include <thread>

io::io(){this -> prefix = "io";}

io::~io(){
    this -> end();
    this -> root_end(); 
    std::map<std::string, TFile*>::iterator itr = this -> files_open.begin(); 
    for (; itr != this -> files_open.end(); ++itr){
        if (!itr -> second){continue;}
        if (itr -> second -> IsOpen()){itr -> second -> Close();}
        itr -> second -> Delete(); 
        this -> pflush(&itr -> second); 
    }
    this -> mflush(&this -> meta_data);
}

void io::import_settings(settings_t* params){
    this -> enable_pyami = params -> fetch_meta; 
    this -> metacache_path = params -> metacache_path; 
    this -> sow_name = params -> sow_name; 
    if (!this -> sow_name.size()){return;}
    this -> info("Checking for Sum of Weights under tree name: " + this -> sow_name); 
}
