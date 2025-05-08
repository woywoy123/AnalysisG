#ifndef STRUCTS_ENUMS_H
#define STRUCTS_ENUMS_H

// ========= (0). add the type here v -- vector, vv -- vector<vector> ============ //
enum class data_enum {
// vector<vector<vector<...>>> -> vvv_<X>
// vector<vector<...>> -> vv_<X>
// vector<...> -> v_<X>
// primitives (float, double, long, ...)
    d  , v_d  , vv_d  , vvv_d  ,
    f  , v_f  , vv_f  , vvv_f  ,
    l  , v_l  , vv_l  , vvv_l  ,
    i  , v_i  , vv_i  , vvv_i  ,
    ull, v_ull, vv_ull, vvv_ull,
    b  , v_b  , vv_b  , vvv_b  ,
    ui , v_ui , vv_ui , vvv_ui ,
    c  , v_c  , vv_c  , vvv_c  , 
    undef, unset // other
}; 
// ================================================================================ //

// optimizers
enum class opt_enum {
    adam, 
    adagrad, 
    adamw, 
    lbfgs, 
    rmsprop, 
    sgd, 
    invalid_optimizer
}; 

enum class mlp_init {
    uniform, 
    normal, 
    xavier_normal,
    xavier_uniform, 
    kaiming_uniform, 
    kaiming_normal
};


// loss functions
enum class loss_enum {
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

enum class scheduler_enum{
    steplr,
    reducelronplateauscheduler,
    lrscheduler,
    invalid_scheduler
};

enum class graph_enum {
    data_graph , data_node , data_edge,
    truth_graph, truth_node, truth_edge,
    edge_index, weight, batch_index, batch_events, 
    pred_graph, pred_node, pred_edge, pred_extra
}; 

enum class mode_enum {
    training, validation, evaluation
}; 

enum class particle_enum {
    index, pdgid, 
    pt, eta, phi, energy, px, pz, py, mass, charge, 
    is_b, is_lep, is_nu, is_add,
    pmc, pmu // bulk cartesian/polar write out
}; 

#endif
