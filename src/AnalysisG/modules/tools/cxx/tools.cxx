#include <tools/tools.h>
tools::tools(){}
tools::~tools(){}

void tools::tflush(std::thread** p){
    if (!*p){return;}
    (*p) -> join(); 
    delete *p; *p = nullptr; 
}

