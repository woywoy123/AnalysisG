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

float plotting::get_max(std::string dim){
    if (dim == "x"){return this -> max(&this -> x_data);}
    if (dim == "y"){return this -> max(&this -> y_data);}
    return 1; 
}

float plotting::get_min(std::string dim){
    if (dim == "x"){return this -> min(&this -> x_data);}
    if (dim == "y"){return this -> min(&this -> y_data);}
    return 1; 
}

float plotting::sum_of_weights(){
    float out = 0; 
    for (size_t x(0); x < this -> weights.size(); ++x){
        out += this -> weights.at(x); 
    }
    if (!out){return 1;}
    return out;
}


