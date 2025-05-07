/**
 * @file cutils_cuda_utils.h
 * @brief Provides utility functions for CUDA operations.
 */

#ifndef CUTILS_CUDA_UTILS_H
#define CUTILS_CUDA_UTILS_H

#include <vector>
#include <torch/torch.h>
#include <c10/cuda/CUDAFunctions.h>
#define _threads 1024

/**
 * @brief Computes the number of blocks needed for a given number of threads.
 *
 * @param lx The total number of threads.
 * @param thl The number of threads per block.
 * @return The number of blocks required.
 */
unsigned int blkn(unsigned int lx, int thl); 

/**
 * @brief Computes the block dimensions for a 1D grid.
 *
 * @param dx The total number of threads in the x-dimension.
 * @param thrx The number of threads per block in the x-dimension.
 * @return The block dimensions for the grid.
 */
const dim3 blk_(unsigned int dx, int thrx); 

/**
 * @brief Computes the block dimensions for a 2D grid.
 *
 * @param dx The total number of threads in the x-dimension.
 * @param thrx The number of threads per block in the x-dimension.
 * @param dy The total number of threads in the y-dimension.
 * @param thry The number of threads per block in the y-dimension.
 * @return The block dimensions for the grid.
 */
const dim3 blk_(unsigned int dx, int thrx, unsigned int dy, int thry); 

/**
 * @brief Computes the block dimensions for a 3D grid.
 *
 * @param dx The total number of threads in the x-dimension.
 * @param thrx The number of threads per block in the x-dimension.
 * @param dy The total number of threads in the y-dimension.
 * @param thry The number of threads per block in the y-dimension.
 * @param dz The total number of threads in the z-dimension.
 * @param thrz The number of threads per block in the z-dimension.
 * @return The block dimensions for the grid.
 */
const dim3 blk_(unsigned int dx, int thrx, unsigned int dy, int thry, unsigned int dz, int thrz); 

/**
 * @brief Changes the device of a tensor.
 *
 * @param inpt Pointer to the input tensor.
 */
void changedev(torch::Tensor* inpt); 

/**
 * @brief Changes the device of a tensor to a specified device.
 *
 * @param dev The target device as a string.
 * @param inx Pointer to the input tensor.
 * @return The tensor on the target device.
 */
torch::Tensor changedev(std::string dev, torch::Tensor* inx); 

/**
 * @brief Creates tensor options based on an input tensor.
 *
 * @param v1 Pointer to the input tensor.
 * @return The tensor options.
 */
torch::TensorOptions MakeOp(torch::Tensor* v1); 

/**
 * @brief Formats a tensor to a specified dimension.
 *
 * @param inpt Pointer to the input tensor.
 * @param dim The target dimensions (default is {-1, 1}).
 * @return The formatted tensor.
 */
torch::Tensor format(torch::Tensor* inpt, std::vector<signed long> dim = {-1, 1}); 

/**
 * @brief Formats a vector of tensors to a specified dimension.
 *
 * @param v The vector of tensors.
 * @param dim The target dimensions (default is {-1, 1}).
 * @return The formatted tensor.
 */
torch::Tensor format(std::vector<torch::Tensor> v, std::vector<signed long> dim = {-1, 1}); 

#endif
