#include <RecursiveGraphNeuralNetwork.h>
#include <transform/cartesian-cuda.h>
#include <physics/cartesian-cuda.h>
#include <transform/polar-cuda.h>
#include <graph/graph-cuda.h>


recursivegraphneuralnetwork::recursivegraphneuralnetwork(int rep, double drop_out){
    this -> drop_out = drop_out; 
    this -> _rep = rep; 

    this -> rnn_dx = new torch::nn::Sequential({
            {"rnn_dx_l1", torch::nn::Linear(this -> _dx + 2*this -> _rep, this -> _rep*2)},
            {"rnn_dx_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep*2}))}, 
            {"rnn_dx_dr1", torch::nn::Dropout(torch::nn::DropoutOptions({this -> drop_out}))},

            {"rnn_dx_r1", torch::nn::SiLU()},
            {"rnn_dx_l2", torch::nn::Linear(this -> _rep*2, this -> _rep*2)},
            {"rnn_dx_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep*2}))}, 
            {"rnn_dx_dr2", torch::nn::Dropout(torch::nn::DropoutOptions({this -> drop_out}))},

            {"rnn_dx_s2", torch::nn::Sigmoid()},
            {"rnn_dx_r2", torch::nn::SiLU()},
            {"rnn_dx_l3", torch::nn::Linear(this -> _rep*2, this -> _rep)}, 
    }); 

    this -> rnn_x = new torch::nn::Sequential({
            {"rnn_x_l1", torch::nn::Linear(this -> _x + this -> _rep, this -> _rep*2)},
            {"rnn_x_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep*2}))}, 
            {"rnn_x_dr1", torch::nn::Dropout(torch::nn::DropoutOptions({this -> drop_out}))},

            {"rnn_x_l2", torch::nn::Linear(this -> _rep*2, this -> _rep*2)},
            {"rnn_x_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep*2}))}, 
            {"rnn_x_dr2", torch::nn::Dropout(torch::nn::DropoutOptions({this -> drop_out}))},

            {"rnn_x_l3", torch::nn::Linear(this -> _rep*2, this -> _rep)}
    }); 

    this -> rnn_merge = new torch::nn::Sequential({
            {"rnn_mrg_l1", torch::nn::Linear(this -> _rep*3, this -> _rep*3)},
            {"rnn_mrg_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep*3}))}, 
            {"rnn_mrg_dr1", torch::nn::Dropout(torch::nn::DropoutOptions({this -> drop_out}))},

            {"rnn_mrg_r1", torch::nn::SiLU()},
            {"rnn_mrg_l2", torch::nn::Linear(this -> _rep*3, this -> _rep)},
            {"rnn_mrg_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep}))}, 
            {"rnn_mrg_dr2", torch::nn::Dropout(torch::nn::DropoutOptions({this -> drop_out}))},
            {"rnn_mrg_l3", torch::nn::Linear(this -> _rep, this -> _rep)}
    }); 

    this -> rnn_update = new torch::nn::Sequential({
            {"rnn_up_l1" , torch::nn::Linear(this -> _output*2 + this -> _rep*2, this -> _rep*2)},
            {"rnn_up_n1" , torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep*2}))}, 
            {"rnn_up_dr1", torch::nn::Dropout(torch::nn::DropoutOptions({this -> drop_out}))},

            {"rnn_up_r1" , torch::nn::SiLU()},
            {"rnn_up_l2" , torch::nn::Linear(this -> _rep*2, this -> _rep*2)},
            {"rnn_up_n2" , torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep*2}))}, 
            {"rnn_up_dr2", torch::nn::Dropout(torch::nn::DropoutOptions({this -> drop_out}))},

            {"rnn_up_l3" , torch::nn::Linear(this -> _rep*2, this -> _output)}
    }); 

    this -> exotic_mlp = new torch::nn::Sequential({
            {"exotic_l1" , torch::nn::Linear(this -> _x + this -> _output, this -> _rep*2)}, 
            {"exotic_n1" , torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep*2}))},
            {"exotic_dr1", torch::nn::Dropout(torch::nn::DropoutOptions({this -> drop_out}))},
            {"exotic_sl1", torch::nn::SiLU()},
            {"exotic_l2" , torch::nn::Linear(this -> _rep*2, this -> _output)}
    });

    this -> node_aggr_mlp = new torch::nn::Sequential({
            {"node_aggr_l1" , torch::nn::Linear(this -> _x, this -> _rep)}, 
            {"node_aggr_n1" , torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep}))},
            {"node_aggr_dr1", torch::nn::Dropout(torch::nn::DropoutOptions({this -> drop_out}))},
            {"node_aggr_l2" , torch::nn::Linear(this -> _rep, this -> _x)}
    }); 

    this -> ntops_mlp = new torch::nn::Sequential({
            {"ntops_l1" , torch::nn::Linear(this -> _x + 4, this -> _rep)}, 
            {"ntops_n1" , torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep}))},
            {"ntops_sl1", torch::nn::SiLU()},

            {"ntops_dr2", torch::nn::Dropout(torch::nn::DropoutOptions({this -> drop_out}))},
            {"ntops_l2" , torch::nn::Linear(this -> _rep, this -> _rep)}, 
            {"ntops_r2", torch::nn::ReLU()},

            {"ntops_l3" , torch::nn::Linear(this -> _rep, this -> _x)}
    }); 

    this -> exo_mlp = new torch::nn::Sequential({
            {"exo_l1", torch::nn::Linear(this -> _x + this -> _output, this -> _x + this -> _output)}, 
            {"exo_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _output + this -> _x}))},
            {"exo_sl1", torch::nn::SiLU()},

            {"exo_dr2", torch::nn::Dropout(torch::nn::DropoutOptions({this -> drop_out}))},
            {"exo_l2" , torch::nn::Linear(this -> _output + this -> _x, this -> _output + this -> _x)}, 
            {"exo_r2", torch::nn::ReLU()},

            {"exo_l3", torch::nn::Linear(this -> _x + this -> _output, this -> _output)}
    }); 

    this -> register_module(this -> rnn_x,         mlp_init::xavier_uniform); 
    this -> register_module(this -> rnn_dx,        mlp_init::xavier_uniform); 
    this -> register_module(this -> rnn_merge,     mlp_init::xavier_uniform); 
    this -> register_module(this -> rnn_update,    mlp_init::xavier_uniform); 
    this -> register_module(this -> node_aggr_mlp, mlp_init::xavier_uniform); 
    this -> register_module(this -> ntops_mlp,     mlp_init::xavier_uniform); 
    this -> register_module(this -> exotic_mlp,    mlp_init::xavier_uniform); 
    this -> register_module(this -> exo_mlp,       mlp_init::xavier_uniform); 
}

torch::Tensor recursivegraphneuralnetwork::message(
        torch::Tensor _trk_i, torch::Tensor _trk_j, 
        torch::Tensor pmc, torch::Tensor pmc_i, torch::Tensor pmc_j,
        torch::Tensor hx_i, torch::Tensor hx_j

){
    torch::Tensor trk_ij = torch::cat({_trk_i, _trk_j}, {-1}); 

    torch::Tensor pmc_i_ = std::get<0>(graph::cuda::unique_aggregation(_trk_i, pmc)); 
    torch::Tensor pmc_j_ = std::get<0>(graph::cuda::unique_aggregation(_trk_j, pmc)); 
    torch::Tensor pmc_ij = std::get<0>(graph::cuda::unique_aggregation(trk_ij, pmc)); 
   
    torch::Tensor m_i  = physics::cuda::cartesian::M(pmc_i);
    torch::Tensor m_j  = physics::cuda::cartesian::M(pmc_j); 
    torch::Tensor m_ij = physics::cuda::cartesian::M(pmc_ij);
   
    torch::Tensor m_i_  = physics::cuda::cartesian::M(pmc_i_);
    torch::Tensor m_j_  = physics::cuda::cartesian::M(pmc_j_); 

    torch::Tensor dr   = physics::cuda::cartesian::DeltaR(pmc_i, pmc_j); 

    std::vector<torch::Tensor> dx_ = {
        m_ij  , pmc_ij, dr, 
        m_i   , m_j    - m_i   , 
        m_i_  , m_j_   - m_i_  , 
        pmc_i , pmc_j  - pmc_i , 
        pmc_i_, pmc_j_ - pmc_i_, 
        hx_i  , hx_j   - hx_i
    }; 

    torch::Tensor hdx = (*this -> rnn_dx) -> forward(torch::cat(dx_, {-1}).to(torch::kFloat32)); 

    torch::Tensor id = torch::cat({hdx, hx_i, hx_j - hx_i}, {-1}); 
    return (*this -> rnn_merge) -> forward(id); 
}

void recursivegraphneuralnetwork::forward(graph_t* data){

    // get the particle 4-vector and convert it to cartesian
    torch::Tensor pt     = data -> get_data_node("pt", this) -> clone();
    torch::Tensor eta    = data -> get_data_node("eta", this) -> clone();
    torch::Tensor phi    = data -> get_data_node("phi", this) -> clone();
    torch::Tensor energy = data -> get_data_node("energy", this) -> clone();
    torch::Tensor pmc    = transform::cuda::PxPyPzE(pt, eta, phi, energy); 

    // the event topology
    torch::Tensor edge_index = data -> get_edge_index(this) -> to(torch::kLong); 
    torch::Tensor num_jets   = data -> get_data_graph("num_jets", this) -> clone(); 
    torch::Tensor num_leps   = data -> get_data_graph("num_leps", this) -> clone(); 
    torch::Tensor met        = data -> get_data_graph("met", this) -> clone(); 
    torch::Tensor met_phi    = data -> get_data_graph("phi", this) -> clone();

    torch::Tensor met_xy  = torch::cat({transform::cuda::Px(met, met_phi), transform::cuda::Py(met, met_phi)}, {-1});
    torch::Tensor src     = edge_index.index({0}).view({-1}); 
    torch::Tensor dst     = edge_index.index({1}).view({-1}); 

    // ------ create an empty buffer for the recursion ------- //
    std::vector<torch::Tensor> x_ = {}; 
    torch::Tensor nulls = torch::zeros_like(pt); 
    for (size_t t(0); t < this -> _rep; ++t){x_.push_back(nulls);}
    nulls = torch::cat(x_, {-1}).to(torch::kFloat32); 

    // ------ Create an initial prediction with the edge classifier ------ //
    nulls = torch::cat({physics::cuda::cartesian::M(pmc), pmc, nulls}, {-1}).to(torch::kFloat32); 
    torch::Tensor hx = (*this -> rnn_x) -> forward(nulls);

    // ------ index the nodes from 0 to N-1 ----- //
    torch::Tensor trk = (torch::ones_like(pt).cumsum({0}) - 1).to(torch::kInt); 

    // ------ index the edges from 0 to N^2 -1 ------ //
    torch::Tensor idx_mlp = (torch::ones_like(src).cumsum({-1})-1).to(torch::kInt); 
    torch::Tensor idx_mat = torch::zeros({trk.size({0}), trk.size({0})}, src.device()).to(torch::kInt); 
    idx_mat.index_put_({src, dst}, idx_mlp); 

    // ------ declare the intermediate states ------- //
    int iter = 0; 
    torch::Tensor H_;
    torch::Tensor edge_index_ = edge_index.clone(); 
    torch::Tensor G_ = torch::zeros_like(torch::cat({src.view({-1, 1}), dst.view({-1, 1})}, {-1})).to(torch::kFloat32); 
    torch::Tensor G  = torch::zeros_like(torch::cat({src.view({-1, 1}), dst.view({-1, 1})}, {-1})).to(torch::kFloat32); 
    while(true){

        // ---- check if the current edge index prediction is null ---- //
        if (!edge_index_.size({1})){break;}

        // ----- get the source and destination indices ---- //
        torch::Tensor src_ = edge_index_.index({0}); 
        torch::Tensor dst_ = edge_index_.index({1}); 

        // ----- Make a new prediction with prior inputs of hx (prior edge prediction) ---- //
        torch::Tensor H = this -> message(
                trk.index({src_}), trk.index({dst_}), 
                pmc, pmc.index({src_}), pmc.index({dst_}), 
                hx.index({src_}), hx.index({dst_})
        ); 

        // ----- if this is the first iteration, initialize H_ ----- //
        if (!iter){H_ = torch::zeros_like(H); ++iter;}

        // ----- use the index matrix to map the source and destination edges to the edge index ----- //
        torch::Tensor idx = idx_mat.index({idx_mlp.index({src_}), idx_mlp.index({dst_})}); 

        // ----- make new prediction of G based on H, H_, prior G, and the current G_ state ---- //
        G  = (*this -> rnn_update) -> forward(torch::cat({H, H_ - H, G, G_.index({idx}) - G}, {-1})); 

        // ----- scale the output by softmax ----- //
        G_.index_put_({idx}, G * G.softmax(-1)); 

        // ----- use new G prediction to get new topology prediction ---- //
        torch::Tensor sel = std::get<1>(G.max({-1})); 

        // ---- check if the new prediction is simply null ---- /
        if (!sel.index({sel == 1}).size({0})){break;}
        G  = G.index({sel != 1}); 
        H_ = H.index({sel != 1}); 

        // ----- update the topology state ---- //
        edge_index_ = edge_index_.index({torch::indexing::Slice(), sel != 1}); 

        // ----- create a new intermediate state of the node ----- //
        std::vector<torch::Tensor> gr_ = graph::cuda::edge_aggregation(edge_index, G_, pmc)["1"]; 

        trk  = gr_[0].index({gr_[2]});
        torch::Tensor pmc_ = gr_[3]; 
        nulls = torch::cat({physics::cuda::cartesian::M(pmc_), pmc_, hx}, {-1}); 
        hx = (*this -> rnn_x) -> forward(nulls.to(torch::kFloat32));
    }

    // ----------- count the number of tops and use them for Z/H boson ---------- //
    std::vector<torch::Tensor> gr_clust = graph::cuda::edge_aggregation(edge_index, G_, pmc)["1"];  

    // ----------- reverse the clustering on a per node basis ----------- //
    torch::Tensor clusters = gr_clust[0].index({gr_clust[2]}); 
    torch::Tensor c_ij  = torch::cat({clusters.index({src}), clusters.index({dst})}, {-1}); 
   
    // ----------- perform a single edge update based on clusters ----------- //
    torch::Tensor z_pmc = std::get<0>(graph::cuda::unique_aggregation(c_ij, pmc));
    torch::Tensor z_inv = physics::cuda::cartesian::M(z_pmc) - this -> res_mass;  
    torch::Tensor z_feat = torch::cat({z_pmc, z_inv, G_}, {-1}); 
    torch::Tensor res_edge = (*this -> exotic_mlp) -> forward(z_feat.to(torch::kFloat32));  

    // ---------- compress edge details to node features ------------ //
    torch::Tensor node_matrix = torch::zeros_like(idx_mat).to(torch::kFloat32); 
    torch::Tensor top_matrix_0 = node_matrix.clone(); 
    torch::Tensor top_matrix_1 = node_matrix.clone(); 
    top_matrix_0.index_put_({src, dst}, G_.index({torch::indexing::Slice(), 0})); 
    top_matrix_1.index_put_({src, dst}, G_.index({torch::indexing::Slice(), 1})); 
    torch::Tensor top_matrix = torch::cat({top_matrix_0.sum({-1}, true), top_matrix_1.sum({-1}, true)}, {-1}); 
    top_matrix = torch::cat({top_matrix, physics::cuda::cartesian::M(gr_clust[3]) - torch::ones_like(pt)*172.62*1000}, {-1}); 
    top_matrix = torch::cat({top_matrix, (clusters > -1).sum({-1}, true), torch::ones_like(pt)*gr_clust[0].size({0})}, {-1}); 
    top_matrix = (*this -> node_aggr_mlp) -> forward(top_matrix.to(torch::kFloat32)); 

    // ---------- compress node details to graph -------- //
    top_matrix = top_matrix.sum({0}, true); 
    top_matrix = torch::cat({top_matrix, num_jets, num_leps, met_xy}, {-1}); 
    torch::Tensor ntops = (*this -> ntops_mlp) -> forward(top_matrix.to(torch::kFloat32));  
    torch::Tensor is_res = (*this -> exo_mlp) -> forward(torch::cat({ntops, (res_edge.softmax(-1)*res_edge).sum({0}, true)}, {-1})); 

    this -> prediction_graph_feature("signal" , is_res); 
    this -> prediction_graph_feature("ntops"  , ntops); 
    this -> prediction_edge_feature("top_edge", G_); 
    this -> prediction_edge_feature("res_edge", res_edge); 

    if (!this -> inference_mode){return;}
    this -> prediction_extra("top_edge_score", G_.softmax(-1));
    this -> prediction_extra("ntops_score"   , ntops.softmax(-1)); 

    this -> prediction_extra("res_edge_score", res_edge.softmax(-1));
    this -> prediction_extra("is_res_score"  , is_res.softmax(-1)); 

    torch::Tensor top_pred = graph::cuda::edge_aggregation(edge_index, G_, pmc)["1"][1]; 
    this -> prediction_extra("top_pmc", top_pred); 

    if (!this -> is_mc){return;}
    torch::Tensor truth_t = data -> get_truth_edge("top_edge", this) -> view({-1}); 
    torch::Tensor truth_ = torch::zeros_like(G_); 
    for (int x(0); x < G_.size({-1}); ++x){truth_.index_put_({truth_t == x, x}, 1);}
    torch::Tensor truth_top = graph::cuda::edge_aggregation(edge_index, truth_, pmc)["1"][1]; 
    this -> prediction_extra("truth_top_pmc", truth_top); 
}

recursivegraphneuralnetwork::~recursivegraphneuralnetwork(){}
model_template* recursivegraphneuralnetwork::clone(){
    recursivegraphneuralnetwork* rnn = new recursivegraphneuralnetwork(this -> _rep, this -> drop_out); 
    rnn -> _dx      = this -> _dx;      
    rnn -> _x       = this -> _x;       
    rnn -> _output  = this -> _output;  
    rnn -> res_mass = this -> res_mass; 
    rnn -> is_mc    = this -> is_mc;
    return rnn;  
}
