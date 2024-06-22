#include <generators/optimizer.h>
#include <thread>
optimizer::optimizer(){
    this -> prefix = "optimizer"; 
    this -> metric = new metrics(); 
}

optimizer::~optimizer(){
    delete this -> metric; 
    std::map<int, model_template*>::iterator itx = this -> kfold_sessions.begin(); 
    for (; itx != this -> kfold_sessions.end(); ++itx){delete itx -> second;}
}

void optimizer::import_dataloader(dataloader* dl){
    this -> metric -> var_pt = this -> var_pt; 
    this -> metric -> var_eta = this -> var_eta;
    this -> metric -> var_phi = this -> var_phi;
    this -> metric -> var_energy = this -> var_energy; 
    this -> metric -> targets = this -> targets; 
    this -> metric -> nbins = this -> nbins; 
    this -> metric -> max_range = this -> max_range; 
    this -> loader = dl;
}

void optimizer::import_model_sessions(std::tuple<model_template*, optimizer_params_t*>* models){
    model_template*       base = std::get<0>(*models); 
    optimizer_params_t* config = std::get<1>(*models); 
    base -> shush = true; 
    base -> set_optimizer(config -> optimizer); 
    
    model_settings_t settings;
    base -> clone_settings(&settings); 
    
    this -> info("_____ IMPORTING MODELS _____"); 
    for (int x(0); x < this -> kfolds; ++x){
        model_template* model_k = base -> clone(); 
        if (x){model_k -> shush = true;}
        model_k -> set_optimizer(config -> optimizer);
        model_k -> import_settings(&settings);
        model_k -> initialize(config); 
        this -> kfold_sessions[x] = model_k; 
    }
}

void optimizer::check_model_sessions(int example_size){
    this -> metric -> epochs = this -> epochs; 
    std::vector<graph_t*> rnd = this -> loader -> get_random(example_size); 
    std::map<int, model_template*>::iterator itx = this -> kfold_sessions.begin(); 
    this -> info("Testing each k-fold model " + std::to_string(rnd.size()) + "-times"); 
    for (; itx != this -> kfold_sessions.end(); ++itx){
        this -> info("____ Checking Model ____: " + std::to_string(itx -> first +1)); 
        for (size_t x(0); x < rnd.size(); ++x){
            if (!x){itx -> second -> shush = false;}
            else {itx -> second -> shush = true;}
            itx -> second -> check_features(rnd[x]);
        } 
        this -> metric -> register_model(itx -> second, itx -> first); 
    }
}

void optimizer::training_loop(int k, int epoch){
    std::vector<graph_t*>* smpl = this -> loader -> get_k_train_set(k); 
    model_template* model = this -> kfold_sessions[k]; 
    model -> epoch = epoch+1; 
    model -> kfold = k+1;
    if (this -> continue_training){model -> restore_state();}

    std::string msg = "training   k-" + std::to_string(k+1) + " progress "; 
    int idx = 0; 
    int l = smpl -> size(); 
    for (int x(0); x < l; ++x, ++idx){
        model -> forward(smpl -> at(x), true);
        this -> metric -> capture(mode_enum::training, k, epoch, l); 
        if (this -> refresh != idx || (k && epoch > 0)){continue;}
        this -> progressbar(float(x+1)/float(l), msg); 
        idx = 0; 
    }
    model -> save_state(); 
    if (k && epoch > 0){return;}
    this -> progressbar(1.0, msg);
    std::cout << std::endl;
}

void optimizer::validation_loop(int k, int epoch){
    std::vector<graph_t*>* smpl = this -> loader -> get_k_validation_set(k); 
    model_template* model = this -> kfold_sessions[k]; 
    std::string msg = "validation k-" + std::to_string(k+1) + " progress "; 
    int idx = 0; 
    int l = smpl -> size(); 
    for (int x(0); x < l; ++x, ++idx){
        model -> forward(smpl -> at(x), false);
        this -> metric -> capture(mode_enum::validation, k, epoch, l); 
        if (this -> refresh != idx || (k && epoch > 0)){continue;}
        this -> progressbar(float(x+1)/float(l), msg); 
        idx = 0; 
    }
    if (k && epoch > 0){return;}
    this -> progressbar(1.0, msg);
    std::cout << std::endl;
}

void optimizer::evaluation_loop(int k, int epoch){
    std::vector<graph_t*>* smpl = this -> loader -> get_test_set(); 
    model_template* model = this -> kfold_sessions[k]; 
    std::string msg = "evaluation k-" + std::to_string(k+1) + " progress "; 
    int idx = 0; 
    int l = smpl -> size(); 
    for (int x(0); x < l; ++x, ++idx){
        model -> forward(smpl -> at(x), false);
        this -> metric -> capture(mode_enum::evaluation, k, epoch, l); 
        if (this -> refresh != idx || (k && epoch > 0)){continue;}
        this -> progressbar(float(x+1)/float(l), msg); 
        idx = 0; 
    }
    if (k && epoch > 0){return;}
    this -> progressbar(1.0, msg);
    std::cout << std::endl;
}

void optimizer::launch_model(){
    auto lamb = [this](int k, int ep){
        if (this -> training){this -> training_loop(k, ep);}
        if (this -> validation){this -> validation_loop(k, ep);}
        if (this -> evaluation){this -> evaluation_loop(k, ep);}
    }; 

    for (int ep(0); ep < this -> epochs; ++ep){
        std::string msg = std::to_string(ep+1) + "/" + std::to_string(this -> epochs); 
        this -> info("___ STARTING EPOCH ___: " + msg); 
        std::vector<std::thread*> th_(this -> kfolds, nullptr); 
        for (int k(0); k < this -> kfolds; ++k){
            if (!ep){lamb(k, ep);}
            else { th_.push_back(new std::thread(lamb, k, ep));}
        }
        for (size_t x(0); x < th_.size(); ++x){
            if (!th_[x]){continue;}
            th_[x] -> join();
            delete th_[x];
            th_[x] = nullptr; 
        }
        this -> metric -> dump_plots(); 
       
    }
}


