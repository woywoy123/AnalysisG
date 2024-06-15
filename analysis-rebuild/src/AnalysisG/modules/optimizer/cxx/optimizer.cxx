#include <generators/optimizer.h>

optimizer::optimizer(){this -> prefix = "optimizer";}

optimizer::~optimizer(){
    std::map<std::string, std::map<int, container_model_t>>::iterator itr; 
    for (itr = this -> k_model.begin(); itr != this -> k_model.end(); ++itr){
        std::map<int, container_model_t>::iterator itk = itr -> second.begin(); 
        for (; itk != itr -> second.end(); ++itk){itk -> second.flush();}
    }
}

void optimizer::define_model(model_template* _model){
    model_settings_t setts; 
    _model -> clone_settings(&setts); 

    std::string name = _model -> name; 
    if (!name.size()){return this -> warning("No model features set.");}
    for (int x(0); x < this -> k_folds.size(); ++x){
        int k = this -> k_folds[x]; 
        if (this -> k_model.count(name)){
            this -> warning("Model already in collection. Skipping");
            continue;
        }
        if (this -> k_model[name].count(k)){continue;}
        this -> k_model[name][k].model = _model -> clone(); 
        this -> k_model[name][k].model -> import_settings(&setts); 
        this -> k_model[name][k].kfold = k; 
    }

    if (this -> k_folds.size()){
        this -> success("Added Model: " + name); 
        return; 
    }

    this -> warning("No k-Folds specified. Assuming k = 1"); 
    this -> k_folds.push_back(1); 
    this -> define_model(_model); 
    return;
}

void optimizer::define_optimizer(std::string name_op){
    std::map<std::string, std::map<int, container_model_t>>::iterator itr; 
    for (itr = this -> k_model.begin(); itr != this -> k_model.end(); ++itr){
        std::map<int, container_model_t>::iterator itk = itr -> second.begin(); 
        for (; itk != itr -> second.end(); ++itk){itk -> second.is_defined(name_op);}
    }
}

void optimizer::create_data_loader(std::vector<graph_template*>* inpt){
    if (this -> loader){this -> loader -> add_to_collection(inpt);}
    else {this -> loader = new dataloader(); this -> create_data_loader(inpt);}
}

void optimizer::start(){
    std::vector<graph_t*> example = this -> loader -> get_random(); 
    std::map<std::string, std::map<int, container_model_t>>::iterator md; 
    for (md = this -> k_model.begin(); md != this -> k_model.end(); ++md){
        for (int k : this -> k_folds){
            for (graph_t* gr : example){
                md -> second[k].check_input(gr);
            }
        }
    }





    for (int x(0); x < this -> epochs; ++x){


    }







}








