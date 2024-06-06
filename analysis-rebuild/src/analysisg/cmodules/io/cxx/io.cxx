#include <io.h>
#include <random>

io::io(){}
io::~io(){
    this -> end();
    std::map<std::string, TFile*>::iterator itr = this -> files_open.begin(); 
    for (; itr != this -> files_open.end(); ++itr){
        if (itr -> second -> IsOpen()){itr -> second -> Close();}
        delete itr -> second;
    }
    this -> end(); 
}

