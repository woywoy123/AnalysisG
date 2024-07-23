#ifndef STRUCTS_MODEL_H
#define STRUCTS_MODEL_H
#include <string>
#include <vector>
#include <map>

// optimizers
enum opt_enum {
    adam, 
    adagrad, 
    adamw, 
    lbfgs, 
    rmsprop, 
    sgd, 
    invalid_optimizer
}; 

enum mlp_init {
    uniform, 
    normal, 
    xavier_normal,
    xavier_uniform, 
    kaiming_uniform, 
    kaiming_normal
}; 

// loss functions
enum loss_enum {
    bce, 
    bce_with_logits, 
    cosine_embedding, 
    cross_entropy, 
    ctc, 
    hinge_embedding, 
    huber, 
    kl_div, 
    l1, 
    margin_ranking, 
    mse, 
    multi_label_margin, 
    multi_label_soft_margin, 
    multi_margin, 
    nll, 
    poisson_nll, 
    smooth_l1, 
    soft_margin, 
    triplet_margin, 
    triplet_margin_with_distance,
    invalid_loss
};

enum graph_enum {
    data_graph , data_node , data_edge, 
    truth_graph, truth_node, truth_edge
}; 

struct model_settings_t {
    opt_enum    e_optim; 
    std::string s_optim; 

    std::string model_name; 
    std::string model_device; 
    std::string model_checkpoint_path; 
    bool inference_mode; 
    bool is_mc; 

    std::map<std::string, std::string> o_graph; 
    std::map<std::string, std::string> o_node; 
    std::map<std::string, std::string> o_edge; 

    std::vector<std::string> i_graph; 
    std::vector<std::string> i_node; 
    std::vector<std::string> i_edge; 
}; 

#endif
