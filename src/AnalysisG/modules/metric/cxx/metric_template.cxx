#include <templates/metric_template.h>
#include <templates/model_template.h>
#include <meta/meta.h>

metric_template::metric_template(){
    this -> name.set_object(this); 
    this -> name.set_setter(this -> set_name); 
    this -> name.set_getter(this -> get_name); 

    this -> run_names.set_object(this); 
    this -> run_names.set_setter(this -> set_run_name); 
    this -> run_names.set_getter(this -> get_run_name); 

    this -> variables.set_object(this); 
    this -> variables.set_setter(this -> set_variables); 
    this -> variables.set_getter(this -> get_variables); 
}

metric_template::~metric_template(){}
metric_template* metric_template::clone(){return new metric_template();}

metric_template* metric_template::clone(int){
    metric_template* mx = this -> clone(); 
    mx -> _var_type     = this -> _var_type;
    mx -> _epoch_kfold  = this -> _epoch_kfold;
    return mx; 
}


std::map<int, torch::TensorOptions*> metric_template::get_devices(){
    std::map<int, bool> devs; 
    std::map<int, torch::TensorOptions*> out = {}; 
    std::map<std::string, model_template*>::iterator itx = this -> lnks.begin();
    for (; itx != this -> lnks.end(); ++itx){
        int dx = itx -> second -> m_option -> device().index(); 
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

    // ............. compute a k-fold to device hash ................... //
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


int metric_template::add_content(
        std::map<std::string, torch::Tensor*>* data, 
        std::vector<varible_t>* content, int index, std::string prefx
){
    std::map<std::string, torch::Tensor*>::iterator itr = data -> begin();
    for (; itr != data -> end(); ++itr, ++index){
        if (!tt){content -> push_back(variable_t());}
        std::string name = prefx + itr -> first; 
        (*content)[index].process(itr -> second, &name, tt); 
    }
    return index; 
}









void metric_template::execute(metric_t* mtx){
    std::map<graph_enum, std::vector<std::string>>* var = mtx -> vars; 

    model_template* mdl = mtx -> mdlx -> clone(1); 
    mdl -> model_checkpoint_path = *mtx -> pth;  
    mdl -> restore_state(); 
    mdl -> inference_mode = false; 

    std::string hx = std::string(this -> hash(std::to_string(mtx -> device) + "+" + std::to_string(mtx -> kfold)));



    std::map<mode_enum, std::vector<graph_t*>*>::iterator itf = this -> hash_bta[hx].begin(); 
    for (; itf != this -> hash_bta[hx].end(); ++itf){
        std::vector<graph_t*>* smpl = itf -> second; 
        for (size_t x(0); x < 2; ++x){
            mdl -> forward(smpl -> at(x)); 
        } 
    }
    

}

void metric_template::define(){
    std::map<int, std::string>::iterator itk; 
    std::map<int, std::map<int, std::string>>::iterator ite; 
    std::map<std::string, std::map<int, std::map<int, std::string>>>::iterator itx;

    for (itx = this -> _epoch_kfold.begin(); itx != this -> _epoch_kfold.end(); ++itx){
        int dev = lnks[itx -> first] -> m_option -> device().index(); 
        ite = this -> _epoch_kfold[itx -> first].begin(); 
        for (; ite != this -> _epoch_kfold[itx -> first].end(); ++ite){
            itk = this -> _epoch_kfold[itx -> first][ite -> first].begin();
            for (; itk != this -> _epoch_kfold[itx -> first][ite -> first].end(); ++itk){
                metric_t* mx = new metric_t(); 
                mx -> kfold = itk -> first; 
                mx -> epoch = ite -> first;
                mx -> device = dev; 
                mx -> pth = &this -> _epoch_kfold[itx -> first][ite -> first][itk -> first]; 
                mx -> mdlx = lnks[itx -> first]; 
                mx -> vars = &this -> _var_type[itx -> first]; 
                this -> execute(mx);
            }
        } 
    }
}
