#include <generators/optimizer.h>

optimizer::optimizer(){
    this -> prefix = "optimizer";
    this -> metric = new metrics(); 
}

optimizer::~optimizer(){
    std::map<std::string, std::map<int, container_model_t>>::iterator itr; 
    for (itr = this -> k_model.begin(); itr != this -> k_model.end(); ++itr){
        std::map<int, container_model_t>::iterator itk = itr -> second.begin(); 
        for (; itk != itr -> second.end(); ++itk){itk -> second.flush();}
    }
    delete this -> metric; 

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

    if (this -> k_folds.size()){return this -> success("Added Model: " + name);}
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

void optimizer::pretest_model(){
    std::vector<graph_t*> example = this -> loader -> get_random(); 
    std::map<std::string, std::map<int, container_model_t>>::iterator md; 
    for (md = this -> k_model.begin(); md != this -> k_model.end(); ++md){
        for (int k : this -> k_folds){
            this -> info("____ Testing Random Entry for k(" + this -> to_string(k) + ")____"); 
            for (int x(0); x < example.size(); ++x){
                this -> info("_____ Test: (" + this -> to_string((x+1)) + "/" + this -> to_string(example.size()) +") _____"); 
                md -> second[k].model  -> check_features(example[x]);
            }
            this -> metric -> register_model(md -> second[k].model, k); 
        }
    }
    this -> success("--------- Completed Model Pre-test ---------- "); 
}

void optimizer::model_loop(std::vector<graph_t*>* data, container_model_t* model_t, int mode, int epoch){
    int l = data -> size();
    model_template* model = model_t -> model; 
    bool flg = false; 
    if (mode == 0){flg = true;}
    else {flg = false;}

    std::string title = ""; 
    if (mode == 0){title = "Training";}
    else if (mode == 1){title = "Validation";}
    else {title = "Evaluation";}

    for (int x(0); x < data -> size(); ++x){
        model -> forward(data -> at(x), flg); 
        this -> progressbar(float(x)/float(l), title); 
    }
    std::cout << std::endl; 
}

void optimizer::start(){
    this -> pretest_model(); 
    int k = this -> max(&this -> k_folds); 
    this -> loader -> generate_test_set(40); 
    this -> loader -> generate_kfold_set(1); 

    for (int ep(0); ep < this -> epochs; ++ep){
        this -> info("----------- Epoch: " + this -> to_string(ep) + " --------------"); 
        for (int k_ : this -> k_folds){
            std::vector<graph_t*>* train = this -> loader -> get_k_train_set(k_);
            std::vector<graph_t*>* valid = this -> loader -> get_k_validation_set(k_);
            std::vector<graph_t*>* test  = this -> loader -> get_test_set();
            std::map<std::string, std::map<int, container_model_t>>::iterator itr; 
            for (itr = this -> k_model.begin(); itr != this -> k_model.end(); ++itr){
                this -> model_loop(train, &this -> k_model[itr -> first][k_], 0, ep); 
                this -> model_loop(valid, &this -> k_model[itr -> first][k_], 1, ep); 
                this -> model_loop(test , &this -> k_model[itr -> first][k_], 2, ep); 
            }
        }
    }
}
