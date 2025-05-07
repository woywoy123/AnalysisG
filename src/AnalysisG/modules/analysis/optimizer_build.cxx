/**
 * @file optimizer_build.cxx
 * @brief Implementation of model session creation and management in the analysis class.
 *
 * This file contains the implementation of methods in the analysis class responsible for
 * building model sessions, configuring optimizers, and managing training progress.
 * It handles the initialization of models, transfer of data to appropriate devices,
 * and provides reporting functionality for monitoring training status.
 */

#include <AnalysisG/analysis.h>
#include <ROOT/RDataFrame.hxx>

/**
 * @brief Builds model sessions for training and inference.
 *
 * This method initializes model sessions based on configuration settings,
 * sets up optimizers, and transfers data to appropriate devices. It prepares
 * the environment for model training or inference by:
 * 1. Creating device maps for data tensors
 * 2. Setting up optimizers with appropriate parameters
 * 3. Configuring k-fold cross-validation
 * 4. Distributing training/validation data appropriately
 */
void analysis::build_model_session(){

    if (!this -> model_sessions.size()){return this -> info("No Models Specified. Skipping.");}
    
    // Configure k-fold cross-validation
    std::vector<int> kfold = this -> m_settings.kfold; 
    if (!kfold.size()){
        // If no specific folds are specified, create all folds from 0 to kfolds-1
        for (int k(0); k < this -> m_settings.kfolds; ++k){
            kfold.push_back(k);
        }
    }
    else {
        // Adjust fold indices (users typically specify folds starting from 1)
        for (size_t k(0); k < kfold.size(); ++k){
            kfold[k] = kfold[k]-1;
        }
    }
    this -> m_settings.kfold = kfold; 

    // Set up device maps for tensor operations
    std::map<int, torch::TensorOptions*> dev_map; 
    for (size_t x(0); x < this -> model_sessions.size(); ++x){
        // Extract device options from the model
        torch::TensorOptions* op = std::get<0>(this -> model_sessions[x]) -> m_option; 
        int dx = op -> device().index();
        
        // Only add each device once to the map
        if (dev_map.count(dx)){continue;}
        dev_map[dx] = op; 
    }
    
    // Transfer data graphs to appropriate devices
    this -> loader -> datatransfer(&dev_map); 

    // Create and configure optimizer for each model session
    for (size_t x(0); x < this -> model_sessions.size(); ++x){
        std::string name = this -> model_session_names[x]; 

        // Initialize optimizer with appropriate settings
        optimizer* optim = new optimizer();
        optim -> m_settings = this -> m_settings; 
        optim -> m_settings.run_name = name; 
        optim -> import_dataloader(this -> loader);
        this -> trainer[name] = optim; 

        // Import model session parameters
        std::tuple<model_template*, optimizer_params_t*> para = this -> model_sessions[x]; 
        optim -> import_model_sessions(&para); 

        // Configure training data for each k-fold
        for (size_t k(0); k < this -> m_settings.kfold.size(); ++k){
            int k_ = this -> m_settings.kfold[k]; 
            std::vector<graph_t*>* check = this -> loader -> get_k_train_set(k_); 
            if (!check){continue;}
            model_report* mx = nullptr;  
            if (this -> m_settings.debug_mode){initialize_loop(optim, k_, std::get<0>(para), std::get<1>(para), &mx);}
            else {this -> threads.push_back(new std::thread(initialize_loop, optim, k_, std::get<0>(para), std::get<1>(para), &mx));}
            while (!mx){std::this_thread::sleep_for(std::chrono::microseconds(10));}
            this -> reports[mx -> run_name + std::to_string(mx -> k)] = mx; 
        }
    }
}

/**
 * @brief Retrieves training progress information for all models.
 * @return A map of model names to vectors of float values representing progress metrics.
 *
 * This method collects progress information from all active model training sessions,
 * including metrics like loss and accuracy values over time.
 */
std::map<std::string, std::vector<float>> analysis::progress(){
    std::map<std::string, std::vector<float>> output; 
    std::map<std::string, model_report*>::iterator itx;
    for (itx = this -> reports.begin(); itx != this -> reports.end(); ++itx){
        model_report* mr = itx -> second; 
        if (!mr -> num_evnt){mr -> num_evnt = 1;}
        mr -> progress = float(mr -> iters) / float(mr -> num_evnt); 
        output[itx -> first] = {itx -> second -> progress*100, float(mr -> iters), float(mr -> num_evnt)}; 
    }
    return output; 
}

/**
 * @brief Retrieves the current training mode for all models.
 * @return A map of model names to strings indicating current training mode.
 *
 * This method provides information about the current operation mode of each
 * training session (e.g., "training", "validation", "inference").
 */
std::map<std::string, std::string> analysis::progress_mode(){
    std::map<std::string, std::string> output; 
    std::map<std::string, model_report*>::iterator itx;
    for (itx = this -> reports.begin(); itx != this -> reports.end(); ++itx){
        std::string o = itx -> second -> mode;
        o += "| k-" + std::to_string(itx -> second -> k+1); 
        o += "| RunName: " + itx -> second -> run_name; 
        o += "| Epoch: " + std::to_string(itx -> second -> epoch);
        output[itx -> first] = o; 
    }
    return output; 
}

/**
 * @brief Retrieves detailed progress reports for all models.
 * @return A map of model names to report strings.
 *
 * This method provides detailed textual progress reports for each training session,
 * including current metrics, epoch information, and time estimates.
 */
std::map<std::string, std::string> analysis::progress_report(){
    std::map<std::string, std::string> output; 
    std::map<std::string, model_report*>::iterator itx;
    for (itx = this -> reports.begin(); itx != this -> reports.end(); ++itx){
        output[itx -> first] = itx -> second -> print(); 
        metrics* plt = itx -> second -> waiting_plot; 
        if (!plt){continue;}
        plt -> dump_plots(itx -> second -> k); 
        itx -> second -> waiting_plot = nullptr;  
    }
    return output; 
}

/**
 * @brief Checks if training is complete for all models.
 * @return A map of model names to boolean values indicating completion status.
 *
 * This method determines whether each model training session has completed
 * its specified number of epochs or met other completion criteria.
 */
std::map<std::string, bool> analysis::is_complete(){
    std::map<std::string, bool> output; 
    std::map<std::string, model_report*>::iterator itx = this -> reports.begin();
    for (; itx != this -> reports.end(); ++itx){output[itx -> first] = itx -> second -> is_complete;}
    return output; 
}

/**
 * @brief Attaches training threads for parallel execution.
 *
 * This method starts training threads for each optimizer, allowing multiple
 * models to train concurrently when appropriate hardware resources are available.
 */
void analysis::attach_threads(){this -> monitor(&this -> threads);}
