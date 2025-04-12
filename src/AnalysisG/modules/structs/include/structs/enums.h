#ifndef STRUCTS_ENUMS_H
#define STRUCTS_ENUMS_H

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
    truth_graph, truth_node, truth_edge,
    edge_index, weight, batch_index, batch_events, 
    pred_graph, pred_node, pred_edge, pred_extra
}; 

enum mode_enum {
    training, validation, evaluation
}; 


#endif
