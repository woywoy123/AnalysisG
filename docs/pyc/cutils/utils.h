#ifndef UTILS_H
#define UTILS_H

#include <torch/torch.h>
#include <vector>

/**
 * @brief Performs a clipping operation on a tensor along a specified dimension.
 * @details This function takes an input tensor and reduces its size or selects a specific
 *          part along the dimension indicated by the `dim` parameter. The exact nature
 *          of the "clipping" (e.g., selecting a sub-tensor, clamping values) depends
 *          on the function's implementation. It operates on the tensor pointed to by `inpt`.
 * @param inpt A pointer to the input `torch::Tensor` that needs to be clipped.
 *             The tensor pointed to might be modified or used as a source.
 * @param dim An integer specifying the dimension along which the clipping operation
 *            should be performed. The validity and interpretation of this dimension index
 *            depend on the input tensor's shape.
 * @return A `torch::Tensor` representing the result of the clipping operation. This might
 *         be a new tensor or a view of the original tensor, depending on the implementation.
 */
torch::Tensor clip(torch::Tensor* inpt, int dim);

/**
 * @brief Consolidates or rearranges a vector of tensors into a single tensor.
 * @details This function takes a vector containing `torch::Tensor` objects and processes
 *          them to produce a single output tensor. The "formatting" could involve operations
 *          like stacking the tensors along a new dimension, concatenating them along an
 *          existing dimension, or interleaving their elements, depending on the specific
 *          implementation requirements. The function operates on the vector pointed to by `inpt`.
 * @param inpt A pointer to a `std::vector<torch::Tensor>`. The tensors within this vector
 *             are the source data for the formatting operation.
 * @return A `torch::Tensor` that represents the formatted combination of the input tensors.
 *         The shape and data of the resulting tensor depend on the specific formatting logic.
 */
torch::Tensor format(std::vector<torch::Tensor>* inpt);

/**
 * @brief Consolidates or rearranges a vector of tensor pointers into a single tensor.
 * @details Similar to the other `format` function, this version processes multiple tensors
 *          to produce a single output tensor. However, it takes a `std::vector` containing
 *          pointers (`torch::Tensor*`) to the input tensors. The formatting operation
 *          (e.g., stacking, concatenation) uses the tensors referenced by these pointers.
 * @param inpt A `std::vector<torch::Tensor*>` containing pointers to the tensors that
 *             need to be formatted.
 * @return A `torch::Tensor` that represents the formatted combination of the tensors
 *         pointed to by the elements in the input vector.
 */
torch::Tensor format(std::vector<torch::Tensor*> inpt);

/**
 * @brief Constructs a `torch::TensorOptions` object based on an existing tensor's properties.
 * @details This utility function inspects the tensor pointed to by `x` to determine its
 *          properties, such as data type (dtype), device (CPU/CUDA), and potentially layout.
 *          It then creates and returns a `torch::TensorOptions` object configured with these
 *          same properties. This is useful for creating new tensors that are compatible
 *          (e.g., on the same device and with the same data type) with an existing tensor.
 * @param x A pointer to a `torch::Tensor` whose properties (dtype, device, etc.) will be
 *          used to configure the returned options object.
 * @return A `torch::TensorOptions` object initialized with the data type and device
 *         (and potentially other properties) of the tensor pointed to by `x`.
 */
torch::TensorOptions MakeOp(torch::Tensor* x);

/**
 * @brief Attempts to change the device of the tensor pointed to by the input pointer.
 * @details This function is intended to move the tensor data associated with the pointer `inpt`
 *          to a different device (e.g., from CPU to GPU or vice-versa). The target device seems
 *          to be implicitly determined (e.g., a default device configured elsewhere).
 *          Note: Standard PyTorch tensor device transfers typically return a *new* tensor.
 *          The `void` return type suggests this function might attempt an in-place modification
 *          (less common for device changes) or modify the pointer `inpt` itself to point to
 *          a new tensor on the target device, or have other side effects related to the tensor's device.
 *          Care should be taken regarding memory management and object lifetime if the pointer is modified.
 * @param inpt A pointer to the `torch::Tensor` whose device placement is intended to be changed.
 *             The tensor object or the pointer itself might be modified.
 */
void changedev(torch::Tensor* inpt);

/**
 * @brief Moves a tensor to a specified device.
 * @details This function takes a pointer to a tensor (`inx`) and a string (`dev`) specifying
 *          the target device (e.g., "cpu", "cuda", "cuda:0"). It creates and returns a
 *          new tensor containing the same data but residing on the specified device. If the
 *          input tensor is already on the target device, it might return the original tensor
 *          or a copy, depending on the implementation. This aligns with the typical behavior
 *          of PyTorch's `.to()` method.
 * @param dev A `std::string` representing the target device for the tensor.
 *            Examples include "cpu", "cuda", "cuda:1".
 * @param inx A pointer to the input `torch::Tensor` that needs to be moved. The original
 *            tensor pointed to by `inx` is typically not modified.
 * @return A `torch::Tensor` residing on the device specified by `dev`. This is often a
 *         new tensor instance, but could be the original tensor if no move was needed.
 */
torch::Tensor changedev(std::string dev, torch::Tensor* inx);

#endif // UTILS_H
