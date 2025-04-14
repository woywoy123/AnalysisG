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

size_t metric_template::size(){
    size_t sx = 0;
    std::map<int, std::map<int, std::string>>::iterator ite; 
    std::map<std::string, std::map<int, std::map<int, std::string>>>::iterator itx;
    for (itx = this -> _epoch_kfold.begin(); itx != this -> _epoch_kfold.end(); ++itx){
        ite = this -> _epoch_kfold[itx -> first].begin(); 
        for (; ite != this -> _epoch_kfold[itx -> first].end(); ++ite){sx += ite -> second.size();} 
    }
    return sx; 
}

std::string enums_to_string(graph_enum gr){
    switch(gr){
        case graph_enum::truth_graph:  return "truth::graph::"; 
        case graph_enum::truth_node :  return "truth::node::"; 
        case graph_enum::truth_edge :  return "truth::edge::"; 
        case graph_enum::pred_graph:   return "prediction::graph::"; 
        case graph_enum::pred_node :   return "prediction::node::"; 
        case graph_enum::pred_edge :   return "prediction::edge::"; 
        case graph_enum::data_graph:   return "data::graph::"; 
        case graph_enum::data_node :   return "data::node::"; 
        case graph_enum::data_edge :   return "data::edge::"; 
        case graph_enum::pred_extra  : return "prediction::extra::"; 
        case graph_enum::edge_index  : return "data::edge::"; 
        case graph_enum::batch_index : return "data::node:"; 
        case graph_enum::weight      : return "data::graph::"; 
        case graph_enum::batch_events: return "data::graph::"; 
        default: return "undef"; 
    }
}

void metric_template::construct(
        std::map<graph_enum, std::vector<variable_t*>>* varx, 
        std::map<graph_enum, std::vector<std::string>>* req, 
        model_template* mdl, graph_t* grx, std::string* mtx
){
    std::map<graph_enum, std::vector<std::string>>::iterator vit;
    bool stx = varx -> size() != req -> size(); 
    if (stx){
        for (vit = req -> begin(); vit != req -> end(); ++vit){
            (*varx)[vit -> first] = std::vector<variable_t*>(vit -> second.size(), nullptr);
        }
    }

    int dv = mdl -> device_index;
    for (vit = req -> begin(); vit != req -> end(); ++vit){
        for (size_t t(0); t < vit -> second.size(); ++t){
            if (!stx && !(*varx)[vit -> first][t]){continue;}
            std::string va_ = vit -> second.at(t); 
            torch::Tensor* tnx = nullptr; 
            switch (vit -> first){
                case graph_enum::truth_graph: tnx = grx -> has_feature(vit -> first, va_, dv); break;
                case graph_enum::truth_node : tnx = grx -> has_feature(vit -> first, va_, dv); break;
                case graph_enum::truth_edge : tnx = grx -> has_feature(vit -> first, va_, dv); break;

                case graph_enum::pred_graph: tnx = mdl -> m_p_graph[va_]; break;
                case graph_enum::pred_node : tnx = mdl -> m_p_node[va_];  break;
                case graph_enum::pred_edge : tnx = mdl -> m_p_edge[va_];  break;

                case graph_enum::data_graph: tnx = grx -> has_feature(vit -> first, va_, dv); break;
                case graph_enum::data_node : tnx = grx -> has_feature(vit -> first, va_, dv); break;
                case graph_enum::data_edge : tnx = grx -> has_feature(vit -> first, va_, dv); break;
                
                case graph_enum::pred_extra  : tnx = mdl -> m_p_undef[va_]; break;
                case graph_enum::edge_index  : tnx = grx -> has_feature(vit -> first, va_, dv); break;
                case graph_enum::weight      : tnx = grx -> has_feature(vit -> first, va_, dv); break;
                case graph_enum::batch_index : tnx = grx -> has_feature(vit -> first, va_, dv); break;
                case graph_enum::batch_events: tnx = grx -> has_feature(vit -> first, va_, dv); break;
                default: break; 
            }
            if (!tnx){
                (*mtx) = "\033[1;31m Could not find: " + enums_to_string(vit -> first) + va_ + " (try enabling inference mode). Skipping... \033[0m"; 
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }

            variable_t* v = (*varx)[vit -> first][t]; 
            if (!v){v = new variable_t();}
            else {v -> flush_buffer();}
            v -> process(tnx, &va_, nullptr); 
            if (!stx){continue;}
            (*mtx) = "\033[1;32m (Found " + enums_to_string(vit -> first) + va_ + ") Typed: " + v -> as_string() + "\033[0m"; 
            std::this_thread::sleep_for(std::chrono::seconds(1));
            (*varx)[vit -> first][t] = v; 
        }
    }
}

void metric_template::execute(metric_t* mtx, metric_template* obj, size_t* prg, std::string* msg){
    std::map<graph_enum, std::vector<std::string>>* var = mtx -> vars; 
    std::map<graph_enum, std::vector<variable_t*>>  vou = {}; 
    mtx -> handl = &vou; 

    (*prg) = 1; 
    std::string mf = (*msg); 
    model_template* mdl = mtx -> mdlx -> clone(1); 
    mdl -> model_checkpoint_path = *mtx -> pth;  
    mdl -> restore_state(); 
    bool init = false; 
    obj -> _outdir += "/epoch-" + std::to_string(mtx -> epoch) + "/"; 
    obj -> create_path(obj -> _outdir); 
    obj -> _outdir += "kfold-" + std::to_string(mtx -> kfold+1) + ".root"; 

    obj -> define_variables(); 
    std::string hx = std::string(this -> hash(std::to_string(mtx -> device) + "+" + std::to_string(mtx -> kfold)));
    std::map<mode_enum, std::vector<graph_t*>*>::iterator itf = this -> hash_bta[hx].begin(); 
    for (; itf != this -> hash_bta[hx].end(); ++itf){
        mtx -> train_mode = itf -> first; 
        std::vector<graph_t*>* smpl = itf -> second; 
        for (size_t x(0); x < smpl -> size(); ++x, ++(*prg)){
            graph_t* gr = smpl -> at(x); 
            mdl -> forward(gr, false); 
            this -> construct(&vou, var, mdl, gr, msg);
            if (!init){
                (*msg) = mf; 
                mtx -> build(); 
                init = true;
            }
            obj -> define_metric(mtx); 
        }
    }
    obj -> end(); 
    delete mdl; 
    delete mtx; 
}

void metric_template::define(std::vector<metric_t*>* vr, std::vector<size_t>* num, std::vector<std::string*>* title, size_t* offset){
    std::map<int, std::string>::iterator itk; 
    std::map<int, std::map<int, std::string>>::iterator ite; 
    std::map<std::string, std::map<int, std::map<int, std::string>>>::iterator itx;

    for (itx = this -> _epoch_kfold.begin(); itx != this -> _epoch_kfold.end(); ++itx){
        int dev = this -> lnks[itx -> first] -> m_option -> device().index(); 
        ite = this -> _epoch_kfold[itx -> first].begin(); 
        for (; ite != this -> _epoch_kfold[itx -> first].end(); ++ite){
            itk = this -> _epoch_kfold[itx -> first][ite -> first].begin();
            for (; itk != this -> _epoch_kfold[itx -> first][ite -> first].end(); ++itk){
                metric_t* mx = new metric_t(); 
                mx -> pth    = &this -> _epoch_kfold[itx -> first][ite -> first][itk -> first]; 
                mx -> vars   = &this -> _var_type[itx -> first]; 
                mx -> mdlx   = this -> lnks[itx -> first]; 
                mx -> kfold  = itk -> first; 
                mx -> epoch  = ite -> first;
                mx -> device = dev; 

                size_t xt = 0; 
                std::string hx = std::string(this -> hash(std::to_string(mx -> device) + "+" + std::to_string(mx -> kfold)));
                std::map<mode_enum, std::vector<graph_t*>*>::iterator itf = this -> hash_bta[hx].begin(); 
                for (; itf != this -> hash_bta[hx].end(); ++itf){xt += itf -> second -> size();}

                (*vr)[*offset] = mx; 
                (*num)[*offset] = xt; 
                std::string til = "Epoch::" + std::to_string(ite -> first); 
                til += "-> K(" + std::to_string(itk -> first+1) + ")"; 
                (*title)[*offset] = new std::string(til);  
                (*offset)++; 
            }
        } 
    }
}
