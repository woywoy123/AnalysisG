#include <metrics/accuracy.h>

accuracy_metric::~accuracy_metric(){}
accuracy_metric* accuracy_metric::clone(){return new accuracy_metric();}
accuracy_metric::accuracy_metric(){this -> name = "accuracy";}

void accuracy_metric::define_variables(){
    this -> register_output("event_accuracy_training"   , "edge", &this -> edge_accuracy); 
    this -> register_output("event_accuracy_validation" , "edge", &this -> edge_accuracy); 
    this -> register_output("event_accuracy_evaluation" , "edge", &this -> edge_accuracy); 

    this -> register_output("event_accuracy_training"   , "ntop_scores", &this -> ntop_scores); 
    this -> register_output("event_accuracy_validation" , "ntop_scores", &this -> ntop_scores); 
    this -> register_output("event_accuracy_evaluation" , "ntop_scores", &this -> ntop_scores); 

    this -> register_output("event_accuracy_training"   , "ntop_truth", &this -> ntop_truth); 
    this -> register_output("event_accuracy_validation" , "ntop_truth", &this -> ntop_truth); 
    this -> register_output("event_accuracy_evaluation" , "ntop_truth", &this -> ntop_truth); 

    this -> register_output("global_accuracy_training"  , "edge", &this -> global_edge_accuracy); 
    this -> register_output("global_accuracy_validation", "edge", &this -> global_edge_accuracy); 
    this -> register_output("global_accuracy_evaluation", "edge", &this -> global_edge_accuracy); 

    this -> register_output("global_accuracy_training"  , "ntops", &this -> ntops_accuracy); 
    this -> register_output("global_accuracy_validation", "ntops", &this -> ntops_accuracy); 
    this -> register_output("global_accuracy_evaluation", "ntops", &this -> ntops_accuracy); 
}

void accuracy_metric::event(){
    this -> write("event_accuracy_" + this -> mode, "ntop_scores", &this -> ntop_scores);
    this -> write("event_accuracy_" + this -> mode, "ntop_truth" , &this -> ntop_truth);
    this -> write("event_accuracy_" + this -> mode, "edge", &this -> edge_accuracy, true);
}

void accuracy_metric::batch(){}
void accuracy_metric::end(){
    this -> ntops_accuracy = std::vector<float>(5, 0);

    for (int x(0); x < 5; ++x){
        int c(0); int f(0); 
        for (int y(0); y < 5; ++y){
            if (x == y){c = this -> ntop_accuracy[x][y];}
            else {f += this -> ntop_accuracy[x][y];}
        }
        if ((f + c) == 0){this -> ntops_accuracy[x] = 1.0;}
        else {this -> ntops_accuracy[x] = float(c) / float(f + c);}
    }
    this -> write("global_accuracy_" + this -> mode,  "edge", &this -> global_edge_accuracy); 
    this -> write("global_accuracy_" + this -> mode, "ntops", &this -> ntops_accuracy, true); 
}


void accuracy_metric::define_metric(metric_t* mtx){
    auto acc  =[](std::map<int, int>* acx){return float((*acx)[1]) / float((*acx)[0] + (*acx)[1]);};
    auto maxv =[](std::vector<float>* acx) -> int {
        int idx = 0; 
        float v = acx -> at(0); 
        for (size_t x(0); x < acx -> size(); ++x){
            if (acx -> at(x) < v){continue;}
            v = acx -> at(x); idx = x; 
        }
        return idx; 
    }; 

    std::vector<long> batch_idx             = mtx -> get<std::vector<long>>( graph_enum::batch_index, "index");   
    std::vector<std::vector<int>> edge_idx  = mtx -> get<std::vector<std::vector<int>>>(  graph_enum::edge_index, "index");
    std::vector<std::vector<float>> edge_sc = mtx -> get<std::vector<std::vector<float>>>(graph_enum::pred_extra, "top_edge_score"); 
    std::vector<std::vector<int>>   edge_tr = mtx -> get<std::vector<std::vector<int>>>(  graph_enum::truth_edge, "top_edge"); 

    std::vector<std::vector<int>>   graph_ntops_tru = mtx -> get<std::vector<std::vector<int>>>(  graph_enum::truth_graph, "ntops"); 
    std::vector<std::vector<float>> graph_ntops_prd = mtx -> get<std::vector<std::vector<float>>>(graph_enum::pred_extra , "ntops_score");

    if (!this -> mode.size()){this -> mode = mtx -> mode();}
    if (this -> mode != mtx -> mode()){this -> end();}
    this -> mode = mtx -> mode(); 

    for (size_t x(0); x < graph_ntops_tru.size(); ++x){
        int n_tp = maxv(&graph_ntops_prd[x]); 
        int n_tt = graph_ntops_tru[x][0]; 
        this -> ntop_accuracy[n_tt][n_tp]++;
    }

    std::map<int, std::map<int, int>> bh_acc = {}; 
    for (size_t x(0); x < edge_sc.size(); ++x){
        long src = edge_idx[0][x]; 
        int ed_t =  edge_tr[x][0]; 
        int ibx = batch_idx[src]; 
        int c = int(int(edge_sc[x][0] <= edge_sc[x][1]) == ed_t);
        this -> _global_edge_accuracy[c]++; 
        bh_acc[ibx][c]++; 
    }
    this -> global_edge_accuracy = acc(&this -> _global_edge_accuracy); 
    std::map<int, std::map<int, int>>::iterator itx; 
    for (itx = bh_acc.begin(); itx != bh_acc.end(); ++itx){
        this -> edge_accuracy = acc(&itx -> second); 
        this -> ntop_scores = graph_ntops_prd[itx -> first];  
        this -> ntop_truth  = graph_ntops_tru[itx -> first][0]; 
        this -> event(); 
    }
}

