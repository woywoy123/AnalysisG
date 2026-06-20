#include <recyclx.h>
#include <utils.h>

recyclx::recyclx(){
    this -> dx_nulls = utils::mzero(1, this -> _xrec).detach();
    this -> te_nulls = utils::mzero(1, this -> _xout).detach(); 
    unsigned long rnx = this -> _xrec; 

    std::vector<NetOps> par_node_rnn = {
        NetOps(network::linear   ,  rnx + this -> _xin),
        NetOps(network::layernorm,  rnx               ),
        NetOps(network::linear   ,  rnx               ),
        NetOps(network::layernorm,  rnx               ),
        NetOps(network::leakyrelu,  0                 ),
        NetOps(network::linear   ,  rnx               )
    }; 
    
    this -> 


}

torch::Tensor recyclx::message(torch::Tensor trk_i,  torch::Tensor trk_j, torch::Tensor pmc){
    return trk_i; 
}

void recyclx::forward(graph_t* data){
    // get the particle 4-vector and convert it to cartesian
    torch::Tensor batch_index  = utils::get_batch(this, data); 
    torch::Tensor event_index  = utils::get_event(this, data); 
    torch::Tensor edge_index   = utils::get_edge(this, data); 
    torch::Tensor src          = utils::get_index(&edge_index, 0); 
    torch::Tensor dst          = utils::get_index(&edge_index, 1); 
    torch::Tensor pmc          = utils::build_pmc(this, data); 

    const std::string key_idx = "cls::1::node-indices"; 
    const std::string key_smx = "cls::1::node-sum"; 
    
    // ------ initialize nulls -------- //
    torch::Tensor null_idx = utils::lzero(&src); 
    torch::Tensor node_rnn = utils::get_index(&this -> dx_nulls, null_idx).clone();
    torch::Tensor edge_rnn = utils::get_index(&this -> dx_nulls, null_idx).clone(); 
    torch::Tensor top_edge = utils::get_index(&this -> te_nulls, null_idx).clone();

    torch::Tensor node_i  = utils::node_idx(&batch_index); 
    torch::Tensor path_s  = utils::get_index(&node_i, src); 
    torch::Tensor path_d  = utils::get_index(&node_i, dst); 
    torch::Tensor invBox  = utils::lzero(&this -> dx_nulls, node_i); 

    //// ------- Build indexing mapping from i -> j, 0 to N^2 - 1 ------ //
    //torch::Tensor idx_mat = this -> build_IDX(data, src, dst); 
    //torch::Tensor deC = idx_mat.index({src, dst}); 

    //torch::Dict<std::string, torch::Tensor> gr_; 
    //while (true){

    //    // Node masking 
    //    torch::Tensor sls = idx_mat.index({src, src}); 
    //    torch::Tensor idf = idx_mat.index({dst, src}); 
    //    torch::Tensor mxk = (sls > -1) * (idf > -1); 
    //    if (this -> break_loop(mxk)){break;}
    //    sls = sls.index({mxk}); idf = idf.index({mxk}); 

    //    // ------ Prior paths 
    //    torch::Tensor path_si = path_s.index({sls}); 
    //    torch::Tensor path_ij = path_d.index({idf}); 

    //    // ------ Node properties
    //    torch::Tensor hx_i  = this -> expand(node_rnn, sls); 
    //    torch::Tensor hx_j  = this -> expand(node_rnn, idf);
    //    torch::Tensor hx_ij = this -> expand(edge_rnn, idf); 

    //    // ------- Generate a message based on current nodes ---------- //
    //    // ------- Check if an echo is being relayed by the current nodes ------- //
    //    torch::Tensor idx =       this -> get_value(top_edge).view({-1, 1}); 
    //    torch::Tensor edx = idx * this -> message(path_si, path_ij, &hx_i, &hx_j, pmc, &hx_ij); 
    //    
    //    // ----- Now direct the echo to the node ----- //
    //    torch::Tensor _src = this -> expand(src, sls); 
    //    torch::Tensor _dst = this -> expand(dst, idf); 

    //    torch::Tensor _idx = idx_mat.index({_src, _dst});      // direction from i -> j | @ edge index 
    //    torch::Tensor _eIr = idx_mat.index({_dst, _dst});      // self edges i.e. j -> j 
    //    torch::Tensor _inc = this -> expand(node_rnn, _eIr);   // receive the message from i -> j | t = 1.
    //                                                           // but need to be careful about gradient

    //    //node_rnn.index_put_({_eIr}, edx - _inc);
    //    // ------ If i -> j is possible, then j -> i should also be possible.
    //    // To test, construct a response/acknowledge message and examine the response strength
    //    // 1. Actually receive and store the incoming message -> note: node_rnn is N[e]^2 x 128
    //    // This is by design, because now edge communication features are no longer being convolved by other incoming messsages
    //    // So each edge can be seen as a "singular" connection rather than a "N x " convolution problem        
    //    torch::Tensor pg = edx - this -> expand(node_rnn, _eIr); 
    //    
    //    // 2. We now apply a signature of the received message [j], using prior information along with our own state.
    //    // This state is defined by compressed messages received previously. We are only testing response at this moment nothing else.
    //    torch::Tensor nId = this -> expand(node_rnn, _eIr); 
    //    torch::Tensor nIx = this -> node_encode(pmc.index({_dst}), node_i.index({dst}).view({-1, 1}), &nId);
    //    
    //    //NOTE: last argument is a message because a real message has a sender, receiver and address. 
    //    //- 1. Return to sender
    //    torch::Tensor msgs = this -> message(path_ij, path_si, &hx_j, &hx_i, pmc, &nIx);  
    //    //- 2. Take recipient response [careful i -> j means j is recipient NOT i]. 
    //    //     Return "msgs" back to recipient [j].
    //    torch::Tensor msgr = this -> message(path_si, path_ij, &hx_i, &hx_j, pmc, &msgs);
    //   
    //    // 3. Autodecoder. Now comes the interesting part. Since we the sender know the contents
    //    //    Of the original message [pmc, node_i, nId] we decode the response [notice if "msgs" was altered this would
    //    //    no longer work because additional information is needed.] thus telling the MLP how distorted as particular 
    //    //    path is. I.e. we should AVOID them.
    //    //    
    //    // 4. Measure the differential between the acknowledged packet and the response.
    //    torch::Tensor rsp = (* this -> rnn_hxx) -> forward(torch::cat({nIx, nIx - msgr
    //    


    //    std::cout << msgr << std::endl; 

    //    abort(); 


    //     
    //    









    //
    //    std::cout << node_rnn << std::endl; 

    //}

    //abort(); 
}

recyclx::~recyclx(){}

model_template* recyclx::clone(){
    recyclx* md   = new recyclx(); 
    md -> is_mc = this -> is_mc; 
    return md; 
}
