/**
 * @file structs_enums.h
 * @brief Defines enumerations used throughout the AnalysisG framework.
 *
 * This file contains the declarations of various enumeration types that are used
 * to define states, types, and options across the AnalysisG framework. These enumerations
 * provide type-safe alternatives to using raw integers or strings for representing
 * discrete sets of values.
 */

#ifndef STRUCTS_ENUMS_H ///< Include guard start for STRUCTS_ENUMS_H.
#define STRUCTS_ENUMS_H ///< Definition of the include guard.

/**
 * @enum data_enum
 * @brief Defines data types and structures for handling various data formats.
 *
 * This enumeration represents different data types that can be processed
 * within the framework, including scalar types and various vector containers.
 * The prefix indicates the dimensionality:
 * - v_* for scalar values or 1D vectors
 * - vv_* for 2D vectors (vectors of vectors)
 * - vvv_* for 3D vectors (vectors of vectors of vectors)
 */
enum class data_enum {
    d,       ///< Double scalar
    v_d,     ///< Vector of doubles
    vv_d,    ///< Vector of vectors of doubles
    vvv_d,   ///< Vector of vectors of vectors of doubles
    f,       ///< Float scalar
    v_f,     ///< Vector of floats
    vv_f,    ///< Vector of vectors of floats
    vvv_f,   ///< Vector of vectors of vectors of floats
    l,       ///< Long scalar
    v_l,     ///< Vector of longs
    vv_l,    ///< Vector of vectors of longs
    vvv_l,   ///< Vector of vectors of vectors of longs
    i,       ///< Integer scalar
    v_i,     ///< Vector of integers
    vv_i,    ///< Vector of vectors of integers
    vvv_i,   ///< Vector of vectors of vectors of integers
    ull,     ///< Unsigned long long scalar
    v_ull,   ///< Vector of unsigned long longs
    vv_ull,  ///< Vector of vectors of unsigned long longs
    vvv_ull, ///< Vector of vectors of vectors of unsigned long longs
    b,       ///< Boolean scalar
    v_b,     ///< Vector of booleans
    vv_b,    ///< Vector of vectors of booleans
    vvv_b,   ///< Vector of vectors of vectors of booleans
    ui,      ///< Unsigned integer scalar
    v_ui,    ///< Vector of unsigned integers
    vv_ui,   ///< Vector of vectors of unsigned integers
    vvv_ui,  ///< Vector of vectors of vectors of unsigned integers
    c,       ///< Character scalar
    v_c,     ///< Vector of characters
    vv_c,    ///< Vector of vectors of characters
    vvv_c,   ///< Vector of vectors of vectors of characters
    undef,   ///< Undefined data type
    unset    ///< Unset data type
};

/**
 * @enum opt_enum
 * @brief Defines available optimizer types for machine learning models.
 *
 * This enumeration represents different optimization algorithms that can be used
 * for training machine learning models within the framework.
 */
enum class opt_enum {
    adam,       ///< Adam optimizer (Adaptive Moment Estimation)
    adagrad,    ///< Adagrad optimizer
    adamw,      ///< AdamW optimizer (Adam with weight decay regularization)
    lbfgs,      ///< Limited-memory Broyden–Fletcher–Goldfarb–Shanno optimizer
    rmsprop,    ///< RMSprop optimizer (Root Mean Square Propagation)
    sgd,        ///< Stochastic Gradient Descent optimizer
    invalid_optimizer ///< Invalid optimizer type
};

/**
 * @enum mlp_init
 * @brief Defines initialization methods for multi-layer perceptron weights.
 *
 * This enumeration represents different strategies for initializing the weights
 * of neural network layers, which can significantly affect training dynamics.
 */
enum class mlp_init {
    uniform,         ///< Uniform initialization
    normal,          ///< Normal (Gaussian) initialization
    xavier_normal,   ///< Xavier/Glorot normal initialization
    xavier_uniform,  ///< Xavier/Glorot uniform initialization
    kaiming_uniform, ///< He/Kaiming uniform initialization
    kaiming_normal   ///< He/Kaiming normal initialization
};

/**
 * @enum loss_enum
 * @brief Defines various loss functions for training machine learning models.
 *
 * This enumeration represents different loss functions that can be used
 * to optimize machine learning models during training.
 */
enum class loss_enum {
    bce,                        ///< Binary Cross Entropy loss
    bce_with_logits,            ///< Binary Cross Entropy with Logits loss
    cosine_embedding,           ///< Cosine Embedding loss
    cross_entropy,              ///< Cross Entropy loss
    ctc,                        ///< Connectionist Temporal Classification loss
    hinge_embedding,            ///< Hinge Embedding loss
    huber,                      ///< Huber loss
    kl_div,                     ///< Kullback-Leibler Divergence loss
    l1,                         ///< L1 loss
    margin_ranking,             ///< Margin Ranking loss
    mse,                        ///< Mean Squared Error loss
    multi_label_margin,         ///< Multi-label Margin loss
    multi_label_soft_margin,    ///< Multi-label Soft Margin loss
    multi_margin,               ///< Multi-margin loss
    nll,                        ///< Negative Log Likelihood loss
    poisson_nll,                ///< Poisson Negative Log Likelihood loss
    smooth_l1,                  ///< Smooth L1 loss
    soft_margin,                ///< Soft Margin loss
    triplet_margin,             ///< Triplet Margin loss
    triplet_margin_with_distance, ///< Triplet Margin with Distance loss
    invalid_loss                ///< Invalid loss type
};

/**
 * @enum graph_enum
 * @brief Defines different components or aspects of a graph structure.
 *
 * This enumeration represents various elements of a graph that can be
 * processed or analyzed separately, such as nodes, edges, or the graph as a whole.
 */
enum class graph_enum {
    data_graph,    ///< Data graph
    data_node,     ///< Data node
    data_edge,     ///< Data edge
    truth_graph,   ///< Ground truth graph
    truth_node,    ///< Ground truth node
    truth_edge,    ///< Ground truth edge
    edge_index,    ///< Edge index
    weight,        ///< Weight
    batch_index,   ///< Batch index
    batch_events,  ///< Batch events
    pred_graph,    ///< Predicted graph
    pred_node,     ///< Predicted node
    pred_edge,     ///< Predicted edge
    pred_extra     ///< Predicted extra data
};

/**
 * @enum mode_enum
 * @brief Defines operational modes for the framework.
 *
 * This enumeration represents different modes in which the framework can operate.
 */
enum class mode_enum {
    training,    ///< Training mode
    validation,  ///< Validation mode
    evaluation   ///< Evaluation mode
};

/**
 * @enum particle_enum
 * @brief Defines properties of particles in a physics simulation.
 *
 * This enumeration represents various attributes of particles that can be
 * used in simulations or analyses.
 */
enum class particle_enum {
    index,    ///< Particle index
    pdgid,    ///< Particle ID
    pt,       ///< Transverse momentum
    eta,      ///< Pseudorapidity
    phi,      ///< Azimuthal angle
    energy,   ///< Energy
    px,       ///< Momentum in x-direction
    pz,       ///< Momentum in z-direction
    py,       ///< Momentum in y-direction
    mass,     ///< Mass
    charge,   ///< Electric charge
    is_b,     ///< Is a b-quark
    is_lep,   ///< Is a lepton
    is_nu,    ///< Is a neutrino
    is_add,   ///< Is an additional particle
    pmc,      ///< Bulk cartesian momentum
    pmu       ///< Bulk polar momentum
};

#endif // STRUCTS_ENUMS_H ///< End of include guard.
