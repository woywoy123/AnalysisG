#include <plotting/plotting.h>
#include <cmath>

plotting::plotting(){}
plotting::~plotting(){   
    std::map<int, std::vector<std::vector<double>>*>::iterator itx;  
    std::map<std::string, std::map<int, std::vector<std::vector<double>>*>>::iterator itm;
    for (itm = this -> roc_data.begin(); itm != this -> roc_data.end(); ++itm){
        for (itx = itm -> second.begin(); itx != itm -> second.end(); ++itx){
            delete this -> roc_data[itm -> first][itx -> first];
            delete this -> labels[itm -> first][itx -> first]; 
        }
    }
}

std::string plotting::build_path(){
    std::string path = this -> output_path;
    if (!this -> ends_with(&path, "/")){path += "/";}
    this -> create_path(path); 
    this -> create_path(path + "raw/"); 
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

std::tuple<float, float> plotting::mean_stdev(std::vector<float>* data){
    float n = data -> size();
    if (n < 2){return {(*data)[0], 0};}
    float mean = 0; 
    for (float k : *data){mean += k;}
    mean = mean / n; 

    float stdev = 0; 
    for (float k : *data){stdev += std::pow(k - mean, 2);}
    return {mean, std::pow(stdev / (n-1), 0.5)}; 
}

void plotting::build_error(){
    if (this -> y_error_down.size()){return;}
    std::map<float, std::vector<float>> maps; 
    for (size_t x(0); x < this -> x_data.size(); ++x){
        float t = this -> x_data[x]; 
        maps[t].push_back(this -> y_data[x]); 
    }

    this -> x_data.clear();
    this -> y_data.clear();

    std::map<float, std::vector<float>>::iterator itr; 
    for (itr = maps.begin(); itr != maps.end(); ++itr){
        std::tuple<float, float> r = this -> mean_stdev(&itr -> second);
        this -> x_data.push_back(itr -> first); 
        this -> y_data.push_back(std::get<0>(r)); 
        float up = std::get<0>(r) + std::get<1>(r); 
        float down = std::get<0>(r) - std::get<1>(r); 
        if (down < 0){down = 0;}
        this -> y_error_down.push_back(down); 
        this -> y_error_up.push_back(up); 
    }

}

void plotting::build_ROC(
        std::string name, int kfold, std::vector<int>* label, std::vector<std::vector<double>>* scores
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

std::vector<roc_t> plotting::get_ROC(){
    std::vector<roc_t> outp; 
    std::map<int, std::vector<std::vector<double>>*>::iterator itx;  
    std::map<std::string, std::map<int, std::vector<std::vector<double>>*>>::iterator itm;
    for (itm = this -> roc_data.begin(); itm != this -> roc_data.end(); ++itm){
        for (itx = itm -> second.begin(); itx != itm -> second.end(); ++itx){
            roc_t rc; 
            rc.kfold  = itx -> first; 
            rc.model  = itm -> first;
            rc.truth  = this -> labels[itm -> first][itx -> first]; 
            rc.scores = this -> roc_data[itm -> first][itx -> first]; 
            rc.cls    = this -> labels[itm -> first][itx -> first] -> at(0).size(); 
            outp.push_back(rc);
        }
    }
    return outp; 
}
