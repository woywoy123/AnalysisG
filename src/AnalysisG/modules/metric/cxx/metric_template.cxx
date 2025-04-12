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

void metric_template::define_metric(){}

void metric_template::construct(
        std::map<graph_enum, std::vector<variable_t*>>* varx, 
        std::map<graph_enum, std::vector<std::string>>* req, 
        model_template* mdl, graph_t* grx
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
            if (stx && !(*varx)[vit -> first][t]){continue;}
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
            if (!tnx){continue;}
            if (!(*varx)[vit -> first][t]){(*varx)[vit -> first][t] = new variable_t();}
            (*varx)[vit -> first][t] -> process(tnx, &va_, nullptr); 
            std::cout << (*varx)[vit -> first][t] -> as_string() << std::endl;
        }
    }
}

void metric_template::execute(metric_t* mtx){
    std::map<graph_enum, std::vector<std::string>>* var = mtx -> vars; 
    std::map<graph_enum, std::vector<variable_t*>>  vou = {}; 
    mtx -> handl = &vou; 

    model_template* mdl = mtx -> mdlx -> clone(1); 
    mdl -> model_checkpoint_path = *mtx -> pth;  
    mdl -> restore_state(); 
    mdl -> inference_mode = false; 
    std::string hx = std::string(this -> hash(std::to_string(mtx -> device) + "+" + std::to_string(mtx -> kfold)));
    std::map<mode_enum, std::vector<graph_t*>*>::iterator itf = this -> hash_bta[hx].begin(); 
    for (; itf != this -> hash_bta[hx].end(); ++itf){
        std::vector<graph_t*>* smpl = itf -> second; 
        for (size_t x(0); x < 10; ++x){
            graph_t* gr = smpl -> at(x); 
            mdl -> forward(gr); 
            this -> construct(&vou, var, mdl, gr);

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
