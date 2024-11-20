#ifndef CUTILS_CUDA_UTILS_H
#define CUTILS_CUDA_UTILS_H

#include <vector>
#include <torch/torch.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "#x must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const dim3 BLOCKS(unsigned int threads, unsigned int dx); 
const dim3 BLOCKS(unsigned int threads, unsigned int dx, unsigned int dy); 
const dim3 BLOCKS(unsigned int threads, unsigned int len, unsigned int dy, unsigned int dx); 

torch::TensorOptions MakeOp(torch::Tensor* v1); 
torch::Tensor format(std::vector<torch::Tensor> v, std::vector<signed long> dim = {-1, 1}); 















#endif
