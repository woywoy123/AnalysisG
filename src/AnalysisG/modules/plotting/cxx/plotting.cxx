#include <plotting/plotting.h>

plotting::plotting(){}
plotting::~plotting(){}

std::string plotting::build_path(){
    std::string path = this -> output_path;
    if (!this -> ends_with(&path, "/")){path += "/";}
    this -> create_path(path); 
    path += this -> filename; 
    path += this -> extension; 
    return path;  
}



