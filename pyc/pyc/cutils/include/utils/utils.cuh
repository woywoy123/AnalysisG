#ifndef CUTILS_CUDA_UTILS_H
#define CUTILS_CUDA_UTILS_H

#include <vector>
#include <torch/torch.h>
#include <c10/cuda/CUDAFunctions.h>
#define _threads 1024

unsigned int blkn(unsigned int lx, int thl); 
const dim3 blk_(unsigned int dx, int thrx); 
const dim3 blk_(unsigned int dx, int thrx, unsigned int dy, int thry); 
const dim3 blk_(unsigned int dx, int thrx, unsigned int dy, int thry, unsigned int dz, int thrz); 

void changedev(torch::Tensor* inpt); 
torch::Tensor changedev(std::string dev, torch::Tensor* inx); 
torch::TensorOptions MakeOp(torch::Tensor* v1); 

torch::Tensor format(torch::Tensor* inpt, std::vector<signed long> dim = {-1, 1}); 
torch::Tensor format(std::vector<torch::Tensor> v, std::vector<signed long> dim = {-1, 1}); 


#endif
