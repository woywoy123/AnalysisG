#include <metrics/accuracy.h>
#include <string>

collector::collector(){}
collector::~collector(){}

cdata_t* collector::get_mode(std::string model, std::string mode, int epoch, int kfold){
    if (mode == "evaluation"){return &this -> model_data[model].evaluation_kfold_data[epoch][kfold];}
    if (mode == "validation"){return &this -> model_data[model].validation_kfold_data[epoch][kfold];}
    if (mode == "training"  ){return &this -> model_data[model].training_kfold_data[epoch][kfold];}
    return nullptr; 
}

void collector::add_ntop_truth(
    std::string mode, std::string model, 
    int epoch, int kfold, int data
){
    cdata_t* ins = this -> get_mode(model, mode, epoch, kfold); 
    ins -> ntops_truth.push_back(data);
}

void collector::add_ntop_edge_accuracy(
    std::string mode, std::string model, 
    int epoch, int kfold, int ntops, double data
){
    cdata_t* ins = this -> get_mode(model, mode, epoch, kfold); 
    ins -> ntop_edge_accuracy[ntops].push_back(data);
}

void collector::add_ntop_scores(
    std::string mode, std::string model, 
    int epoch, int kfold, std::vector<double>* data
){
    cdata_t* ins = this -> get_mode(model, mode, epoch, kfold); 
    ins -> ntop_score.push_back(*data);
}

void collector::add_ntru_ntop_scores(
    std::string mode, std::string model, 
    int epoch, int kfold, int ntru, int ntop, double data
){
    cdata_t* ins = this -> get_mode(model, mode, epoch, kfold); 
    ins -> ntru_npred_matrix[ntru][ntop].push_back(data);
}

std::map<std::string, std::vector<cdata_t*>> collector::get_plts(){
    auto lamb = [this](
            std::string base, std::string mode, cmodel_t* in, 
            std::map<std::string, std::vector<cdata_t*>>* out
    ) -> void {
        std::map<int, std::map<int, cdata_t>>* cn = nullptr;
        if (mode == "evaluation"){cn = &in -> evaluation_kfold_data;}
        if (mode == "validation"){cn = &in -> validation_kfold_data;}
        if (mode == "training"  ){cn = &in -> training_kfold_data;  }
        this -> modes.push_back(mode); 
        this -> model_names.push_back(base);
        std::map<int, std::map<int, cdata_t>>::iterator ite = cn -> begin();
        for (; ite != cn -> end(); ++ite){
            std::map<int, cdata_t>::iterator itk = ite -> second.begin(); 
            for (; itk != ite -> second.end(); ++itk){
                std::string key = base + "::" + mode; 
                key += "::epoch-" + std::to_string(ite -> first); 
                itk -> second.kfold = itk -> first; 
                (*out)[key].push_back(&itk -> second); 
                this -> epochs.push_back(ite -> first); 
                this -> kfolds.push_back(itk -> first); 
            }
        }
    }; 

    std::map<std::string, std::vector<cdata_t*>> out = {}; 
    std::map<std::string, cmodel_t>::iterator itx; 
    for (itx = this -> model_data.begin(); itx != this -> model_data.end(); ++itx){
        std::string base_name = itx -> first; 
        lamb(base_name, "training", &itx -> second, &out); 
        lamb(base_name, "validation", &itx -> second, &out); 
        lamb(base_name, "evaluation", &itx -> second, &out); 
    }

    std::vector<std::string> tmp_m = this -> model_names; 
    std::vector<std::string> tmp_e = this -> modes; 
    std::vector<int> tmp_ep = this -> epochs; 
    std::vector<int> tmp_kf = this -> kfolds; 

    this -> model_names.clear(); 
    this -> modes.clear(); 
    this -> epochs.clear(); 
    this -> kfolds.clear(); 

    this -> unique_key(&tmp_m , &this -> model_names);
    this -> unique_key(&tmp_e , &this -> modes); 
    this -> unique_key(&tmp_ep, &this -> epochs); 
    this -> unique_key(&tmp_kf, &this -> kfolds);  
    return out; 
}






