/**
 * @file utils.cuh
 * @brief This file contains utility functions designed to facilitate common operations
 *        within CUDA and PyTorch C++ environments, particularly focusing on kernel launch
 *        configurations and tensor manipulations.
 */

#ifndef CUTILS_CUDA_UTILS_H
#define CUTILS_CUDA_UTILS_H

#include <vector>
#include <torch/torch.h>
#include <c10/cuda/CUDAFunctions.h>

/**
 * @def _threads
 * @brief Defines a default number of threads per block for CUDA kernels.
 * @details This macro provides a convenient constant, typically set to a power of 2
 *          like 1024, which is often a reasonable default for thread block sizes on
 *          many NVIDIA GPUs. It can be used when specifying the block dimensions in
 *          kernel launches.
 */
#define _threads 1024

/**
 * @brief Calculates the required number of blocks for a CUDA kernel launch.
 * @details This function determines the number of blocks needed in a grid dimension
 *          to cover a total number of elements or threads (`lx`), given a specific
 *          number of threads per block (`thl`) in that dimension. It typically uses
 *          integer division with ceiling rounding (`(lx + thl - 1) / thl`) to ensure
 *          that all elements are processed, even if `lx` is not perfectly divisible
 *          by `thl`. This is essential for configuring the grid size in kernel launches.
 * @param lx The total number of elements or threads to be processed in a specific dimension.
 * @param thl The number of threads allocated per block in that same dimension.
 * @return The calculated number of blocks required for the grid in that dimension.
 */
unsigned int blkn(unsigned int lx, int thl);

/**
 * @brief Creates a 1-dimensional `dim3` structure for CUDA kernel launch grid configuration.
 * @details This function simplifies the creation of a `dim3` object, which is used by CUDA
 *          to define the dimensions of the execution grid. This specific overload configures
 *          a 1D grid. It calculates the necessary number of blocks in the x-dimension using
 *          the `blkn` function based on the total size `dx` and the threads per block `thrx`.
 *          The y and z dimensions are implicitly set to 1.
 * @param dx The total size (e.g., number of elements or threads) required in the x-dimension.
 * @param thrx The number of threads per block configured for the x-dimension.
 * @return A `dim3` object representing the grid dimensions (blocks.x, 1, 1).
 */
const dim3 blk_(unsigned int dx, int thrx);

/**
 * @brief Creates a 2-dimensional `dim3` structure for CUDA kernel launch grid configuration.
 * @details This function creates a `dim3` object for configuring a 2D execution grid in CUDA.
 *          It calculates the necessary number of blocks in both the x and y dimensions using
 *          the `blkn` function, based on the total sizes (`dx`, `dy`) and the threads per
 *          block (`thrx`, `thry`) in each respective dimension. The z dimension is implicitly
 *          set to 1.
 * @param dx The total size (e.g., number of elements or threads) required in the x-dimension.
 * @param thrx The number of threads per block configured for the x-dimension.
 * @param dy The total size (e.g., number of elements or threads) required in the y-dimension.
 * @param thry The number of threads per block configured for the y-dimension.
 * @return A `dim3` object representing the grid dimensions (blocks.x, blocks.y, 1).
 */
const dim3 blk_(unsigned int dx, int thrx, unsigned int dy, int thry);

/**
 * @brief Creates a 3-dimensional `dim3` structure for CUDA kernel launch grid configuration.
 * @details This function creates a `dim3` object for configuring a 3D execution grid in CUDA.
 *          It calculates the necessary number of blocks in the x, y, and z dimensions using
 *          the `blkn` function, based on the total sizes (`dx`, `dy`, `dz`) and the threads
 *          per block (`thrx`, `thry`, `thrz`) in each respective dimension.
 * @param dx The total size (e.g., number of elements or threads) required in the x-dimension.
 * @param thrx The number of threads per block configured for the x-dimension.
 * @param dy The total size (e.g., number of elements or threads) required in the y-dimension.
 * @param thry The number of threads per block configured for the y-dimension.
 * @param dz The total size (e.g., number of elements or threads) required in the z-dimension.
 * @param thrz The number of threads per block configured for the z-dimension.
 * @return A `dim3` object representing the grid dimensions (blocks.x, blocks.y, blocks.z).
 */
const dim3 blk_(unsigned int dx, int thrx, unsigned int dy, int thry, unsigned int dz, int thrz);

/**
 * @brief Moves the provided tensor to the default CUDA device in-place.
 * @details This function modifies the input tensor directly, changing its underlying storage
 *          to reside on the currently active CUDA device as determined by the PyTorch/CUDA runtime
 *          (e.g., via `c10::cuda::current_device()`). The tensor's data is copied to the GPU memory.
 *          Use this function when you want to modify the original tensor object.
 * @param inpt A pointer to the `torch::Tensor` object that needs to be moved to the default CUDA device.
 *             The tensor pointed to by `inpt` will be modified.
 */
void changedev(torch::Tensor* inpt);

/**
 * @brief Creates a new tensor on a specified device by copying an existing tensor.
 * @details This function takes an input tensor (`inx`) and creates a *new* tensor containing
 *          the same data but residing on the device specified by the `dev` string (e.g., "cuda:0",
 *          "cuda:1", "cpu"). The original tensor pointed to by `inx` remains unchanged on its
 *          original device.
 * @param dev A string specifying the target device for the new tensor (e.g., "cuda:0", "cpu").
 * @param inx A pointer to the source `torch::Tensor` whose data will be copied. This tensor is not modified.
 * @return A new `torch::Tensor` object located on the specified device `dev`, containing a copy
 *         of the data from the tensor pointed to by `inx`.
 */
torch::Tensor changedev(std::string dev, torch::Tensor* inx);

/**
 * @brief Creates a `torch::TensorOptions` object based on the properties of an existing tensor.
 * @details This utility function inspects the properties (like data type, device, layout) of the
 *          input tensor `v1` and constructs a `torch::TensorOptions` object reflecting these properties.
 *          This is useful when you need to create new tensors that should have the same characteristics
 *          (e.g., dtype and device) as an existing tensor, ensuring consistency.
 * @param v1 A pointer to the `torch::Tensor` from which to derive the options. This tensor is not modified.
 * @return A `torch::TensorOptions` object configured with the data type, device, layout, etc.,
 *         matching the tensor pointed to by `v1`.
 */
torch::TensorOptions MakeOp(torch::Tensor* v1);

/**
 * @brief Reshapes a tensor to the specified dimensions.
 * @details This function takes a tensor and changes its shape according to the provided `dim` vector.
 *          It likely utilizes PyTorch's `view` or `reshape` operations internally. The total number
 *          of elements must remain consistent between the original tensor and the target shape.
 *          One dimension in `dim` can be specified as -1, in which case its size will be inferred
 *          based on the total number of elements and the sizes of the other dimensions.
 * @param inpt A pointer to the `torch::Tensor` to be reshaped. Depending on the internal implementation
 *             (view vs. reshape), the original tensor's data might be shared or copied.
 * @param dim A `std::vector<signed long>` defining the desired shape for the output tensor.
 * @return A `torch::Tensor` (potentially a view or a new tensor) with the shape specified by `dim`.
 */
torch::Tensor format(torch::Tensor* inpt, std::vector<signed long> dim);

/**
 * @brief Concatenates a vector of tensors and reshapes the result.
 * @details This function first concatenates all tensors within the input vector `v` along a default
 *          dimension (usually dimension 0). Then, it reshapes the resulting single tensor to the
 *          shape specified by the `dim` vector. This is useful for combining multiple tensors
 *          (e.g., batches or segments) into a single tensor with a specific target layout.
 * @param v A `std::vector` containing `torch::Tensor` objects to be concatenated.
 * @param dim A `std::vector<signed long>` defining the desired final shape for the concatenated tensor.
 *            One dimension can be -1 for size inference.
 * @return A single `torch::Tensor` containing the concatenated data from the input vector `v`,
 *         reshaped according to the `dim` specification.
 */
torch::Tensor format(std::vector<torch::Tensor> v, std::vector<signed long> dim);

#endif // CUTILS_CUDA_UTILS_H
