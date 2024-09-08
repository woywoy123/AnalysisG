#include <generators/optimizer.h>

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
    this -> metric -> m_settings = this -> m_settings; 
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
    for (int x(0); x < this -> m_settings.kfold.size(); ++x){
        int k_ = this -> m_settings.kfold[x]; 
        std::vector<graph_t*>* check = this -> loader -> get_k_train_set(k_); 
        if (!check){continue;}
 
        model_template* model_k = base -> clone(); 
        if (x){model_k -> shush = true;}
        model_k -> set_optimizer(config -> optimizer);
        model_k -> import_settings(&settings);
        model_k -> initialize(config); 
        this -> kfold_sessions[k_] = model_k; 
    }
}

void optimizer::check_model_sessions(int example_size, std::map<std::string, model_report*>* rep){
    std::vector<graph_t*> rnd = this -> loader -> get_random(example_size); 
    this -> info("Testing each k-fold model " + std::to_string(rnd.size()) + "-times");

    std::map<int, model_template*>::iterator itx = this -> kfold_sessions.begin(); 
    for (; itx != this -> kfold_sessions.end(); ++itx){
        std::string msg = "____ Checking Model ____:"; 
        msg += " kfold -> " + std::to_string(itx -> first +1); 
        msg += " RunName -> " + this -> m_settings.run_name; 
        this -> info(msg); 
        for (size_t x(0); x < rnd.size(); ++x){
            if (!x){itx -> second -> shush = false;}
            else {itx -> second -> shush = true;}
            itx -> second -> check_features(rnd[x]);
        } 
        model_report* mr = this -> metric -> register_model(itx -> second, itx -> first); 
        this -> reports[mr -> run_name + std::to_string(mr -> k)] = mr; 
        (*rep)[mr -> run_name + std::to_string(mr -> k)] = mr; 
    }
}

void optimizer::training_loop(int k, int epoch){
    std::vector<graph_t*>* smpl = this -> loader -> get_k_train_set(k); 
    model_template* model = this -> kfold_sessions[k]; 

    int l = smpl -> size(); 
    model_report* mr = this -> reports[this -> m_settings.run_name + std::to_string(k)]; 
    mr -> mode = "training";
    mr -> epoch = epoch+1;  
    if (this -> m_settings.batch_size > 1){
        std::vector<std::vector<graph_t*>> batched = this -> discretize(smpl, this -> m_settings.batch_size); 
        l = batched.size();  
        for (int x(0); x < l; ++x){
            model -> forward(batched[x], true);
            this -> metric -> capture(mode_enum::training, k, epoch, l); 
            mr -> progress = float(x+1)/float(l); 
        }
    }
    else {
        for (int x(0); x < l; ++x){
            graph_t* gr = (*smpl)[x]; 
            gr -> in_use = 2; 
            model -> forward(gr, true);
            gr -> in_use = 1; 
            mr -> progress = float(x+1)/float(l); 
            this -> metric -> capture(mode_enum::training, k, epoch, l); 
        }
    }
    model -> save_state(); 
}

void optimizer::validation_loop(int k, int epoch){
    std::vector<graph_t*>* smpl = this -> loader -> get_k_validation_set(k); 
    model_template* model = this -> kfold_sessions[k]; 
    model -> evaluation_mode(true); 

    model_report* mr = this -> reports[this -> m_settings.run_name + std::to_string(k)]; 
    mr -> mode = "validation"; 

    int l = smpl -> size(); 
    for (int x(0); x < l; ++x){
        graph_t* gr = (*smpl)[x]; 
        gr -> in_use = 2; 
        model -> forward(gr, false);
        this -> metric -> capture(mode_enum::validation, k, epoch, l); 
        mr -> progress = float(x+1)/float(l);
        gr -> in_use = 1; 
    }
}

void optimizer::evaluation_loop(int k, int epoch){
    std::vector<graph_t*>* smpl = this -> loader -> get_test_set(); 
    model_template* model = this -> kfold_sessions[k]; 
    model -> evaluation_mode(true); 

    model_report* mr = this -> reports[this -> m_settings.run_name + std::to_string(k)]; 
    mr -> mode = "evaluation"; 

    int l = smpl -> size(); 
    for (int x(0); x < l; ++x){
        graph_t* gr = (*smpl)[x]; 
        gr -> in_use = 2; 
        model -> forward(gr, false);
        this -> metric -> capture(mode_enum::evaluation, k, epoch, l); 
        mr -> progress = float(x+1)/float(l); 
        gr -> in_use = 1; 
    }
}

void optimizer::launch_model(int k){
    auto lamb = [this](int k, int ep){
        model_template* model = this -> kfold_sessions[k]; 
        model -> epoch = ep+1; 
        model -> kfold = k+1;

        if (!this -> m_settings.continue_training){}
        else if (model -> restore_state()){return;}

        if (this -> m_settings.training){this -> training_loop(k, ep);}
        if (this -> m_settings.validation){this -> validation_loop(k, ep);}
        if (this -> m_settings.evaluation){this -> evaluation_loop(k, ep);}

        if (this -> m_settings.debug_mode){return this -> metric -> dump_plots(k);}

        model_report* mr = this -> reports[this -> m_settings.run_name + std::to_string(k)]; 
        mr -> waiting_plot = this -> metric; 
        while (mr -> waiting_plot){std::this_thread::sleep_for(std::chrono::milliseconds(10));}
        //c10::cuda::CUDACachingAllocator::emptyCache();  
    }; 

    for (int ep(0); ep < this -> m_settings.epochs; ++ep){lamb(k, ep);}
    model_report* mr = this -> reports[this -> m_settings.run_name + std::to_string(k)]; 
    mr -> is_complete = true; 
}

