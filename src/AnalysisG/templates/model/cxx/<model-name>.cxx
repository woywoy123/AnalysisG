#include <<model-name>.h>

<model-name>::<model-name>(){

    this -> example = new torch::nn::Sequential({
            {"L1", torch::nn::Linear(2, 2)},
            {"RELU", torch::nn::ReLU()},
            {"L2", torch::nn::Linear(2, 2)}
    }); 

    this -> register_module(this -> example); 
}

void <model-name>::forward(graph_t* data){

    // fetch the input data of the model.
    // If the variable is not available, this will return a nullptr.
    // Note: the second argument is always `this` (selects the correct device tensor).
    torch::Tensor graph = data -> get_data_graph("graph", this) -> clone(); 
    torch::Tensor node  = data -> get_data_node("node", this) -> clone();
    torch::Tensor edge  = data -> get_data_edge("edge", this) -> clone(); 
    torch::Tensor edge_index = data -> get_edge_index(this) -> clone(); 

    // output the prediction weights for edges, nodes, graphs.
    this -> prediction_graph_feature("...", <some-tensor>); 
    this -> prediction_node_feature("...", <some-tensor>);
    this -> prediction_edge_feature("...", <some-tensor>); 
    if (!this -> inference_mode){return;} // skips any variables not available during inference time.
    this -> prediction_extra("...", <some-tensor>);  // Any variables that should be dumped during the inference.
}

<model-name>::~<model-name>(){}
model_template* <model-name>::clone(){
    return new <model-name>(); 
}
