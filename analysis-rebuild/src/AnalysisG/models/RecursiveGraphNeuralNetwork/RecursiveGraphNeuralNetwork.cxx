#include <RecursiveGraphNeuralNetwork.h>

recursivegraphneuralnetwork::recursivegraphneuralnetwork(){
    this -> rnn_dx = new torch::nn::Sequential({
            {"rnn_dx_l1", torch::nn::Linear(this -> _dx*2, this -> _hidden)},
            {"rnn_dx_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_dx_l2", torch::nn::Linear(this -> _hidden, this -> _repeat)}
    }); 

    this -> register_module(this -> rnn_dx); 


}

recursivegraphneuralnetwork::~recursivegraphneuralnetwork(){}

model_template* recursivegraphneuralnetwork::clone(){
    return new recursivegraphneuralnetwork(); 
}
