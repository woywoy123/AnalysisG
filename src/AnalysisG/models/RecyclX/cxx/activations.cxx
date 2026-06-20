#include <pyc/pyc.h>
#include <utils.h>

//torch::nn::Linear    utils::make_FX(long int src, long int dst = -1){
//    return torch::nn::Linear(torch::nn::LinearOptions(src, (dst < 0) ? src : dst).bias(true)); 
//}
//
//torch::nn::LayerNorm utils::make_FX(long int src, long int dst = -1){
//    return torch::nn::LayerNorm(torch::nn::LayerNormOptions({src})); 
//}
//
//torch::nn::PReLU     utils::make_FX(long int n, long int  dst = -1){
//    return torch::nn::PReLU(torch::nn::PReLUOptions().num_parameters(n)); 
//}
//
//torch::nn::Dropout  utils::make_FX(long int n, long int  dst = -1){
//    return torch::nn::Dropout(torch::nn::DropoutOptions({n})); 
//}
//
//
//torch::nn::ReLU      utils::make_FX(long int n, long int dst){return torch::nn::ReLU();}
//torch::nn::SiLU      utils::make_FX(long int n, long int dst){return torch::nn::SiLU();}
//torch::nn::Sigmoid   utils::make_FX(long int n, long int dst){return torch::nn::Sigmoid();}
//
//
