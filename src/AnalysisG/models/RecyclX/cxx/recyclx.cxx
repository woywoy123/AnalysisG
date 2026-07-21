#include <recyclx.h>
#include <utils.h>
#include <pyc/pyc.h>

recyclx::recyclx(){
    this -> dx_nulls = utils::mzero(1, this -> _xrec).detach();
    this -> te_nulls = utils::mzero(1, this -> _xout).detach(); 
    unsigned long rnx = this -> _xrec; 

    std::vector<NetOps> par_node_rnn = {
        NetOps(network::linear   , rnx + this -> _xin),
        NetOps(network::leakyrelu, rnx + this -> _xin),
//        NetOps(network::layernorm, rnx + this -> _xin     ),
        NetOps(network::linear   , rnx + this -> _xin, rnx),
//        NetOps(network::layernorm, rnx, rnx),
        NetOps(network::leakyrelu, rnx, rnx),
        NetOps(network::linear   , rnx, rnx)
    }; 
    this -> rnn_x = utils::make_Network("node", par_node_rnn); 

    std::vector<NetOps> par_msg_rnn = {
        NetOps(network::linear  ,  rnx + this -> _xin),
       NetOps(network::leakyrelu,  rnx + this -> _xin),
//        NetOps(network::layernorm,  rnx + this -> _xin     ),
        NetOps(network::linear   ,  rnx + this -> _xin, rnx),
//        NetOps(network::layernorm,  rnx                    ),
        NetOps(network::leakyrelu,  rnx),
        NetOps(network::linear   ,  rnx)
    }; 
    this -> rnn_hxx = utils::make_Network("msg", par_msg_rnn); 
    
     std::vector<NetOps> par_dec_rnn = {
        NetOps(network::linear   ,  rnx * 2 + this -> _xin         ),
        NetOps(network::leakyrelu,  rnx * 2 + this -> _xin         ),
        NetOps(network::linear   ,  rnx * 2 + this -> _xin, rnx * 2),
        NetOps(network::leakyrelu,  rnx * 2                        ),
        NetOps(network::linear   ,  rnx * 2, 4                     )
    };                                                    
    this -> echo_code = utils::make_Network("decode", par_dec_rnn); 

     std::vector<NetOps> par_top_rnn = {
        NetOps(network::linear   ,  rnx * 2 + this -> _xin         ),
        NetOps(network::leakyrelu,  rnx * 2 + this -> _xin         ),
        NetOps(network::linear   ,  rnx * 2 + this -> _xin, rnx * 2),
        NetOps(network::leakyrelu,  rnx * 2                        ),
        NetOps(network::linear   ,  rnx * 2, this -> _xout         )
    }; 
    this -> rnn_top_edge = utils::make_Network("top_edge", par_top_rnn); 

    this -> register_module(this -> rnn_x  );
    this -> register_module(this -> rnn_hxx);
//    this -> register_module(this -> autodec);
    this -> register_module(this -> rnn_top_edge);

}


void recyclx::forward(graph_t* data){
    
    // get the particle 4-vector and convert it to cartesian
    torch::Tensor batch_index  = utils::get_batch(this, data); 
    torch::Tensor event_index  = utils::get_event(this, data); 
    torch::Tensor edge_index   = utils::get_edge(this, data); 
    torch::Tensor src          = utils::get_index(&edge_index, 0); 
    torch::Tensor dst          = utils::get_index(&edge_index, 1); 
    torch::Tensor pmc          = utils::build_pmc(this, data); 
 
    // ------ initialize null tensors -------- //
    torch::Tensor null_idx = utils::lzero(&src);  // -> [1 x N^2]

    torch::Tensor node_rnn = utils::get_index(&this -> dx_nulls, null_idx); // -> [N^2 x 128]
    torch::Tensor edge_rnn = utils::get_index(&this -> dx_nulls, null_idx); // -> [N^2 x 128]
    torch::Tensor top_edge = utils::get_index(&this -> te_nulls, null_idx); // -> [N^2 x 2  ]

    // --------- 0.0: Define the node indexing, path_state (source) and path_state (destination)
    torch::Tensor node_i  = utils::node_idx(&batch_index);  // Create 0, 1, 2, 3... N
    torch::Tensor path_s  = utils::get_index(&node_i, src); // -> [N^2, src] 
    torch::Tensor path_d  = utils::get_index(&node_i, dst); // -> [N^2, dst]

    
    // --------- 1.0: Encode the current node state --------- //
    torch::Tensor Nxi = utils::NRecode(this, pmc, path_s, &node_rnn); // -> Nxi [N^2, 128]

    // --------- 1.1: Send the encoded state -------- //
    // -> j received msg from i @ node channel fx( src + dst, Nxi)
    torch::Tensor Edi = utils::Message(this, path_s, path_d, pmc, &Nxi);  

    // --------- 1.2: Edi: Normalize against a reference signal -------- //
    // This could be for example the self loops of the node 
    // against the initial node state Nxi
    torch::Tensor Invj = utils::get_index(&Nxi, dst); // -> [N^2, 128]
    torch::Tensor  Edj = utils::get_index(&Edi, dst); // -> [N^2, 128]
   



    std::cout << torch::matmul( Invj, torch::transpose(Invj - Edj, 0, 1) ) << std::endl;
    abort();
//
//
//
//
//
//
//
//
//
//
//    torch::Tensor dEij = utils::get_diff(Edjj, Edi);  // -> [N^2, 2 x 128]
//    
//    // --------- 1.3: Normalizing ------- //
//    // Now we know Edjj denotes a self loop, but we also have Nxi which 
//    // is the original node state of j. So now we compare the incoming 
//    // self message against the original node enocding.
//    torch::Tensor Ndjj = utils::get_diff(Edjj, Nxi);  // -> [N^2, 2 x 128]
//
//    // --------- 1.4: Ping and Echoing -------- //
//    // Now we image we are at node "j" we received a message 
//    // but we dont know its contents, only "i" knows this 
//    // So send the message back to sender.
//    // -> i received msg from i @ node channel.
//    torch::Tensor Edj = utils::Message(this, path_d, path_s, pmc, &Edi); // -> [N^2, 128]
//
//    // -> How much was Nxi distorted through this operation
//    torch::Tensor Dxij = utils::get_diff(Edj, Nxi); // -> [N^2, 2 x 128]
//    torch::Tensor Dxji = utils::get_diff(Edi, Nxi); // -> [N^2, 2 x 128]
//     
//    // --------- 1.5: Quantify the echo signal  ------ // 
//    torch::Tensor Sxij = Ndjj - (Dxij + dEij);  // -> [N^2, 2 x 128]
//    torch::Tensor Sxjj = Ndjj - (Dxji + dEij);  // -> [N^2, 2 x 128]
//   
//    // --------- 1.6: Decode the echo ---------- //
//    torch::Tensor pmcIJ = utils::NDecode(this, path_d, path_s, pmc, &Sxij); // -> [N^2, xin]
//    torch::Tensor pmcJJ = utils::NDecode(this, path_d, path_d, pmc, &Sxjj); // -> [N^2, xin]
//
//    edge_rnn = utils::Message(this, path_d, path_s, pmcIJ, &Nxi); 
//    node_rnn = utils::Message(this, path_d, path_s, pmcJJ, &Nxi); 
//
//    edge_rnn = utils::get_diff(node_rnn, edge_rnn); 
//    top_edge = utils::TopEdge(this, path_s, path_d, pmc, &edge_rnn); 
//    torch::Dict<std::string, torch::Tensor> aggr = pyc::graph::edge_aggregation(edge_index, top_edge, pmc); 
//    torch::Tensor node_idn = aggr.at("cls::1::node-indices").index({node_i}); 
//    torch::Tensor node_sdc = aggr.at("cls::1::node-sum").index({node_i});  
//
    abort(); 


//    this -> prediction_edge_feature("top_edge", top_edge); 
}

recyclx::~recyclx(){}

model_template* recyclx::clone(){
    recyclx* md   = new recyclx(); 
    md -> is_mc = this -> is_mc; 
    return md; 
}
