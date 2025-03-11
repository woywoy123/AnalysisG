#ifndef UTILS_H
#define UTILS_H

#include <torch/torch.h>
#include <vector>

torch::Tensor clip(torch::Tensor* inpt, int dim); 
torch::Tensor format(std::vector<torch::Tensor>* inpt); 
torch::Tensor format(std::vector<torch::Tensor*> inpt); 
torch::TensorOptions MakeOp(torch::Tensor* x); 
void changedev(torch::Tensor* inpt); 
torch::Tensor changedev(std::string dev, torch::Tensor* inx); 

#endif
