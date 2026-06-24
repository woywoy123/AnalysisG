#include <templates/model_template.h>
#include <templates/metric_template.h>
#include <structs/switchboards.h>

metric_t::metric_t(){}
metric_t::~metric_t(){
    if (!this -> handl){return;}
    std::map<graph_enum, std::vector<variable_t*>>::iterator itv; 
    for (itv = this -> handl -> begin(); itv != this -> handl -> end(); ++itv){
        for (size_t x(0); x < itv -> second.size(); ++x){
            if (!itv -> second[x]){continue;}
            itv -> second[x] -> clear = true; 
            delete itv -> second[x]; 
            itv -> second[x] = nullptr; 
        }
    }
    delete this -> handl; 
    if (this -> vars){delete this -> vars;}
}


void metric_t::import_mapping(std::map<graph_enum, std::vector<std::string>> mapping){
    tools::pflush(&this -> vars);  
    this -> vars = new std::map<graph_enum, std::vector<std::string>>(mapping); 
    this -> build(); 
} 

void metric_t::import_model(model_template* _mdl){
    this -> mdlx = _mdl; 
    this -> device = _mdl -> m_option -> device().index();  
    this -> mdlx -> inference_mode = true; 
}


void metric_t::import_graphs(std::vector<graph_t*>* grs){
    this -> batch_graphs = grs; 
    this -> nx = 0; this -> ny = grs -> size(); 
}


void metric_t::build(){
    if (!this -> vars){this -> warning("Need mapping for vars"); return;}
    if (!this -> handl){this -> handl = new std::map< graph_enum, std::vector< variable_t*> >();}
     
    std::map<graph_enum, std::vector<std::string>>::iterator it;
    for (it = this -> vars -> begin(); it != this -> vars -> end(); ++it){
        size_t lx = (*this -> vars)[it -> first].size(); 
        size_t ly = (*this -> handl)[it -> first].size(); 
        if ( lx != ly ){(*this ->  handl)[it -> first] = std::vector<variable_t*>(lx, nullptr);}
        for (size_t x(0); x < it -> second.size(); ++x){ 
            if (!(*this -> handl)[it -> first][x]){(*this -> handl)[it -> first][x] = new variable_t();}
            this -> v_maps[it -> first][it -> second[x]] = x;  
            this -> h_maps[it -> first][it -> second[x]] = true;  
        }
    }
}

std::string* metric_t::get_filename(long unsigned int idx){
    if (!this -> batch_files){return nullptr;}
    if (this -> batch_files -> size() < idx){return nullptr;}
    return this -> batch_files -> at(idx); 
}

std::string metric_t::mode(){return model_mode(this -> _mode);}

void metric_t::getPrediction(){
    if (!this -> vars){
        this -> warning("Provide the output mapping");
        this -> warning("std::map<graph_enum, std::vector<std::string>>"); 
        return; 
    }

    size_t dv = this -> device;
    std::map<graph_enum, std::map<std::string, size_t> >::iterator vit;
    for (vit = this -> v_maps.begin(); vit != this -> v_maps.end(); ++vit){
        std::map<std::string, size_t>::iterator itx; 
        graph_enum grm  = vit -> first; 
        for (itx = this -> v_maps[grm].begin(); itx != this -> v_maps[grm].end(); ++itx){
            variable_t* vx_ = (*this -> handl)[grm][itx -> second]; 
            std::string va_ = itx -> first; 
            torch::Tensor* tnx = nullptr; 
            switch (grm){
                case graph_enum::truth_graph: tnx = this -> gr_i -> has_feature(grm, va_, dv); break;
                case graph_enum::truth_node : tnx = this -> gr_i -> has_feature(grm, va_, dv); break;
                case graph_enum::truth_edge : tnx = this -> gr_i -> has_feature(grm, va_, dv); break;

                case graph_enum::pred_graph: tnx = this -> mdlx -> m_p_graph[va_]; break;
                case graph_enum::pred_node : tnx = this -> mdlx -> m_p_node[va_];  break;
                case graph_enum::pred_edge : tnx = this -> mdlx -> m_p_edge[va_];  break;
                case graph_enum::pred_extra: tnx = this -> mdlx -> m_p_undef[va_]; break;

                case graph_enum::data_graph: tnx = this -> gr_i -> has_feature(grm, va_, dv); break;
                case graph_enum::data_node : tnx = this -> gr_i -> has_feature(grm, va_, dv); break;
                case graph_enum::data_edge : tnx = this -> gr_i -> has_feature(grm, va_, dv); break;
                
                case graph_enum::edge_index  : tnx = this -> gr_i -> has_feature(grm, va_, dv); break;
                case graph_enum::weight      : tnx = this -> gr_i -> has_feature(grm, va_, dv); break;
                case graph_enum::batch_index : tnx = this -> gr_i -> has_feature(grm, va_, dv); break;
                case graph_enum::batch_events: tnx = this -> gr_i -> has_feature(grm, va_, dv); break;
                default: break; 
            }
            if (!tnx){
                std::string mx = "\033[1;31m Could not find: "; 
                mx += enums_to_string(vit -> first) + va_; 
                this -> warning(mx + " (try enabling inference mode). Skipping... \033[0m"); 
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            vx_ -> flush_buffer(); 
            vx_ -> process(tnx, &va_, nullptr); 
        }
    }
}

bool metric_t::next(){
    if (!this -> batch_graphs   ){return false;}
    if (this -> nx >= this -> ny){return false;}
    this -> gr_i        =  this -> batch_graphs -> at(this -> nx);
    this -> batch_files = &this -> gr_i -> batched_filenames; 
    if (this -> mdlx){this -> mdlx -> forward(this -> gr_i, false);}
    this -> getPrediction(); 
    this -> index = this -> nx; 
    this -> nx++; 
    return true; 
}







metric_model_t::metric_model_t(){this -> metric = new metric_t();}

metric_model_t::~metric_model_t(){
    this -> pflush(&this -> metric);
    this -> pflush(&this -> metrx); 
}

bool metric_model_t::verify(){
    if (!this -> model ){return false;}
    if (!this -> metrx ){return false;}
    if (!this -> dev   ){return false;}
    if (!this -> metric){return false;}
    if (!this -> run_name.size()){return false;}
    if (!this -> checkpoint_path.size()){return false;}
    if (this -> kfold < 0){return false;}
    if (this -> epoch < 0){return false;}
    return true; 
}



