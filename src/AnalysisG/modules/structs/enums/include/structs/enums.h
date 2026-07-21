#ifndef ANALYSISG_STRUCTS_ENUMS_H
#define ANALYSISG_STRUCTS_ENUMS_H

#include <string>
#include <map>

namespace AnalysisG {
    namespace enums {
        enum class data {
        // ========= (0). add the type here v -- vector, vv -- vector<vector> ============ //
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
            undefined, unset // other
        }; 
        // ================================================================================ //

        enum class particle {
            index, 
            pdgid, 
            pt, 
            eta, 
            phi, 
            energy, 
            px, 
            pz, 
            py, 
            mass, 
            pmc, 
            pmu, // bulk cartesian/polar write out
            is_b, 
            is_lep, 
            is_nu, 
            is_add,
            charge, 
        }; 

        // optimizers
        enum class optimizers {
            adam, 
            adagrad, 
            adamw, 
            lbfgs, 
            rmsprop, 
            sgd, 
            invalid
        }; 

        enum class samplers {
            uniform, 
            normal, 
            xavier_normal,
            xavier_uniform, 
            kaiming_uniform, 
            kaiming_normal,
            invalid
        };

        // loss functions
        enum class loss {
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
            invalid
        };

        enum class scheduler {
            steplr,
            reducelronplateauscheduler,
            lrscheduler,
            invalid
        };

        enum class graph {
            data_graph, truth_graph, pred_graph, extra_graph,
            data_node , truth_node , pred_node , extra_node ,
            data_edge , truth_edge , pred_edge , extra_edge ,
            weight    , batch_index, batch_events, 
            pred_extra, edge_index , invalid
        }; 
        
        enum class mode {
            training  , 
            validation, 
            evaluation, 
            invalid
        }; 

        enum class network {
            linear, 
            layernorm, 
            dropout,
            relu,
            silu, 
            sigmoid, 
            prelu,
            leakyrelu, 
            tanh, 
            invalid
        }; 

        // -------- Translation layers --------- //
        std::string as_string(AnalysisG::enums::graph gr); 
        std::string as_string(AnalysisG::enums::mode st); 
        std::string as_string(AnalysisG::enums::data rt); 

        AnalysisG::enums::graph  as_graph(const std::string* val); 
        AnalysisG::enums::mode   as_mode(const std::string* val); 
        AnalysisG::enums::data   as_data(const std::string* val); 

        std::map<AnalysisG::enums::mode, std::string> as_mode(const std::map<std::string, std::string>* val); 
        // -------------------------------------- //

    }
}

#endif
