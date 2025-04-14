#include <templates/metric_template.h>
#include <templates/model_template.h>

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
    if (this -> hash_bta.count(hsx) && this -> hash_bta[hsx].count(mx)){return;}
    this -> hash_bta[hsx][mx] = data;
}


