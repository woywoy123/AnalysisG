#include <io.h>
#include <random>

io::io(){this -> prefix = "io";}
io::~io(){
    this -> end();
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

