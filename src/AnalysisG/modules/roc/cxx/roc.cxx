#include <plotting/roc.h>
#include <cmath>

roc::roc(){}
roc::~roc(){
    std::map<int, std::vector<std::vector<double>>*>::iterator itx;  
    std::map<std::string, std::map<int, std::vector<std::vector<double>>*>>::iterator itm;
    for (itm = this -> roc_data.begin(); itm != this -> roc_data.end(); ++itm){
        for (itx = itm -> second.begin(); itx != itm -> second.end(); ++itx){
            delete this -> roc_data[itm -> first][itx -> first];
            delete this -> labels[itm -> first][itx -> first]; 
        }
    }
    for (size_t x(0); x < this -> ptr_roc.size(); ++x){delete this -> ptr_roc[x];}
}

void roc::build_ROC(
        std::string name, int kfold, 
        std::vector<int>* label, 
        std::vector<std::vector<double>>* scores
){
    size_t clx = (scores) ? scores -> at(0).size() : (1 + this -> max(label));
    size_t lsx = (scores) ? scores -> size() : label -> size();

    if (this -> roc_data[name].count(kfold)){
        if (scores){
            delete this -> roc_data[name][kfold];
            this -> roc_data[name][kfold] = this -> generate<double>(lsx, clx);
        }
    }
    else if (!scores){this -> roc_data[name][kfold] = this -> generate<double>(lsx, clx);}
    else {this -> roc_data[name][kfold] = this -> generate<double>(lsx, clx);}


    if (this -> labels[name].count(kfold)){
        if (label){
            delete this -> labels[name][kfold];
            this -> labels[name][kfold] = this -> generate<int>(lsx, clx);
        }
    }
    else if (!label){this -> labels[name][kfold] = this -> generate<int>(lsx, clx);}
    else {this -> labels[name][kfold] = this -> generate<int>(lsx, clx);}

    std::vector<std::vector<double>>* vs = this -> roc_data[name][kfold]; 
    std::vector<std::vector<int   >>* vt = this -> labels[name][kfold]; 

    for (size_t x(0); x < lsx; ++x){
        for (size_t y(0); y < clx*(1 - !scores); ++y){(*vs)[x][y] = scores -> at(x).at(y);}
        for (size_t y(0); y < clx*(1 - !label ); ++y){(*vt)[x][y] = label  -> at(x) == y;}
    }
}

std::vector<roc_t*> roc::get_ROC(){
    std::map<int, std::vector<std::vector<double>>*>::iterator itx;  
    std::map<std::string, std::map<int, std::vector<std::vector<double>>*>>::iterator itm;
    for (itm = this -> roc_data.begin(); itm != this -> roc_data.end(); ++itm){
        for (itx = itm -> second.begin(); itx != itm -> second.end(); ++itx){
            roc_t* rc = new roc_t(); 
            rc -> kfold  = itx -> first; 
            rc -> model  = itm -> first;
            rc -> truth  = this -> labels[itm -> first][itx -> first]; 
            rc -> scores = this -> roc_data[itm -> first][itx -> first]; 
            rc -> cls    = this -> labels[itm -> first][itx -> first] -> at(0).size(); 
            this -> ptr_roc.push_back(rc);
        }
    }
    return this -> ptr_roc; 
}

