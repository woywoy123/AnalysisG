#include <templates/metric_template.h>
#include <templates/model_template.h>

std::map<int, torch::TensorOptions*> metric_template::get_devices(){
    std::map<int, bool> devs; 
    std::map<int, torch::TensorOptions*> out = {}; 
    std::map<std::string, model_template*>::iterator itx = this -> lnks.begin();
    for (; itx != this -> lnks.end(); ++itx){
        int dx = itx -> second -> device_index; 
        if (devs[dx]){continue;}
        devs[dx] = true; 
        out[dx] = itx -> second -> m_option; 
    }
    return out;
}

std::vector<int> metric_template::get_kfolds(){
    std::map<int, bool> kdx; 
    std::vector<int> out = {}; 

    std::map<std::string, std::map<int, std::map<int, std::string>>>::iterator itx;
    for (itx = this -> _epoch_kfold.begin(); itx != this -> _epoch_kfold.end(); ++itx){
        std::map<int, std::map<int, std::string>>::iterator ite = itx -> second.begin(); 
        for (; ite != itx -> second.end(); ++ite){
            std::map<int, std::string>::iterator itk = ite -> second.begin();
            for (; itk != ite -> second.end(); ++itk){
                int k = itk -> first;
                if (kdx[k]){continue;}
                out.push_back(k); 
                kdx[k] = true; 
            }
        } 
    }
    return out;
}


bool metric_template::link(model_template* mdl){
    std::string mdlx = mdl -> name; 
    if (this -> lnks.count(mdlx)){return true;}
    bool ok = true; 
    if (!this -> _var_type.count(mdlx)){
        this -> failure("Invalid Variable ModelName. Got " + mdlx + " expected:"); 
        std::map<std::string, std::map<graph_enum, std::vector<std::string>>>::iterator itx; 
        for (itx = this -> _var_type.begin(); itx != this -> _var_type.end(); ++itx){this -> failure("Model: " + itx -> first);}
        ok = false; 
    }

    if (!this -> _epoch_kfold.count(mdlx)){
        this -> failure("Invalid RunName ModelName. Got " + mdlx + " expected:"); 
        std::map<std::string, std::map<int, std::map<int, std::string>>>::iterator itx; 
        for (itx = this -> _epoch_kfold.begin(); itx != this -> _epoch_kfold.end(); ++itx){this -> failure("Model: " + itx -> first);}
        ok = false; 
    }
    this -> info("Linking Model: " + mdlx + " to metric: " + std::string(this -> name)); 
    if (!ok){return ok;}
    this -> lnks[mdlx] = mdl; 

    // ............. compute k-fold to device hash ................... //
    std::string dx_hx = std::to_string(mdl -> m_option -> device().index()) + "+"; 
    std::map<int, std::map<int, std::string>>::iterator ite = this -> _epoch_kfold[mdlx].begin(); 
    for (; ite != this -> _epoch_kfold[mdlx].end(); ++ite){
        std::map<int, std::string>::iterator itk = ite -> second.begin();
        for (; itk != ite -> second.end(); ++itk){
            std::string hx = this -> hash(dx_hx + std::to_string(itk -> first)); 
            if (this -> hash_mdl.count(hx)){continue;}
            this -> hash_mdl[hx].push_back(mdl); 
        } 
    }
    return ok; 
}


void metric_template::link(std::string hsx, std::vector<graph_t*>* data, mode_enum mx){
    if (this -> hash_bta.count(hsx)){return;}
    this -> hash_bta[hsx][mx] = data; 
}


