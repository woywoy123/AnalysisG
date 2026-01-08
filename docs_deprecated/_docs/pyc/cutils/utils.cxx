/**
 * @file utils.cxx
 * @brief This file implements utility functions primarily focused on manipulating PyTorch tensors within a C++ environment.
 *
 * It provides functionalities for tensor slicing (clipping), concatenation (formatting),
 * device management, and creating tensor options based on existing tensors. These utilities
 * aim to simplify common tensor operations often required in machine learning pipelines
 * implemented using the LibTorch (PyTorch C++) library.
 */

#include <torch/torch.h>
#include <vector>
#include <string>

/**
 * @brief Extracts a slice from a tensor along a specified dimension, effectively "clipping" it.
 *
 * This function performs a slicing operation on the input tensor. Specifically, it selects
 * the data corresponding to the index 0 along the dimension specified by `dim`. This is
 * often used to isolate the first element or feature map along a particular axis.
 * For example, if `dim` is 1, it returns `inpt->select(1, 0)`.
 *
 * @param inpt A pointer to the source `torch::Tensor` from which the slice will be extracted.
 *             The function expects `inpt` to point to a valid tensor object. The tensor
 *             pointed to by `inpt` is not modified by this function. Behavior is undefined
 *             if `inpt` is a null pointer.
 * @param dim The integer index of the dimension along which to perform the clipping (slicing).
 *            This dimension index must be valid for the input tensor's shape (i.e., within
 *            the range `[-inpt->dim(), inpt->dim() - 1]`). Providing an invalid dimension
 *            will likely result in a runtime error from the underlying PyTorch library.
 *
 * @return torch::Tensor A new `torch::Tensor` representing the slice. This returned tensor
 *         will typically be a *view* of the original tensor's data (sharing the underlying
 *         memory) but with a reduced dimensionality (the specified `dim` is removed).
 *         Its shape will be the same as the input tensor's shape, but without the `dim`-th
 *         dimension.
 *
 * @throws std::runtime_error (or similar PyTorch-specific exception) May be thrown if the
 *         provided `dim` is out of bounds for the input tensor.
 *
 * @note The term "clip" here refers to selecting a specific slice (index 0), not value clipping
 *       (like `torch::clamp`).
 */
torch::Tensor clip(torch::Tensor* inpt, int dim);

/**
 * @brief Concatenates a vector of tensors along their last dimension.
 *
 * This function takes a vector of `torch::Tensor` objects and joins them together into a
 * single, larger tensor. The concatenation happens along the last dimension (i.e., `dim = -1`).
 * All tensors in the input vector must have compatible shapes for concatenation: they must
 * have the same number of dimensions, and their sizes must match in all dimensions except
 * for the last one.
 *
 * @param inpt A pointer to a `std::vector<torch::Tensor>`. The vector contains the tensors
 *             to be concatenated. The function expects `inpt` to point to a valid vector.
 *             The vector itself and the tensors it contains are not modified. Behavior is
 *             undefined if `inpt` is a null pointer. If the vector is empty, the behavior
 *             might depend on the underlying `torch::cat` implementation (potentially
 *             returning an empty tensor or throwing an error).
 *
 * @return torch::Tensor A new `torch::Tensor` that is the result of concatenating all tensors
 *         in the input vector along their last dimension. The resulting tensor's data type
 *         and device will be determined by PyTorch's type promotion and device rules,
 *         typically matching the first tensor in the vector if all are consistent.
 *
 * @throws std::runtime_error (or similar PyTorch-specific exception) May be thrown if the
 *         input tensors have incompatible shapes for concatenation or if the input vector
 *         is empty and `torch::cat` requires at least one tensor.
 *
 * @note This function operates on a vector of tensor *objects*. See the overload taking
 *       `std::vector<torch::Tensor*>` for handling vectors of tensor pointers.
 */
torch::Tensor format(std::vector<torch::Tensor>* inpt);

/**
 * @brief Concatenates tensors pointed to by a vector of tensor pointers along their last dimension.
 *
 * This function takes a vector of pointers to `torch::Tensor` objects and joins the pointed-to
 * tensors together into a single, larger tensor. The concatenation happens along the last
 * dimension (i.e., `dim = -1`). All tensors pointed to by the elements in the input vector
 * must have compatible shapes for concatenation: they must have the same number of dimensions,
 * and their sizes must match in all dimensions except for the last one.
 *
 * @param inpt A `std::vector<torch::Tensor*>` containing pointers to the tensors to be
 *             concatenated. The function iterates through the vector, dereferences each pointer,
 *             and uses the resulting tensor in the concatenation. The vector itself is passed
 *             by value (though it contains pointers), and the tensors pointed to are not modified.
 *             The caller is responsible for ensuring that all pointers in the vector are valid
 *             (non-null) and point to initialized tensor objects. Behavior is undefined if
 *             the vector contains null pointers. If the vector is empty, the behavior might
 *             depend on the underlying `torch::cat` implementation.
 *
 * @return torch::Tensor A new `torch::Tensor` that is the result of concatenating all tensors
 *         pointed to by the elements of the input vector along their last dimension. The
 *         resulting tensor's data type and device will be determined by PyTorch's type
 *         promotion and device rules.
 *
 * @throws std::runtime_error (or similar PyTorch-specific exception) May be thrown if the
 *         pointed-to tensors have incompatible shapes, if any pointer in the vector is null,
 *         or if the input vector is empty and `torch::cat` requires at least one tensor.
 *
 * @note This function operates on a vector of tensor *pointers*. Ensure proper memory management
 *       and validity of the pointers passed in the vector.
 */
torch::Tensor format(std::vector<torch::Tensor*> inpt);

/**
 * @brief Creates a `torch::TensorOptions` object configured with the data type and device of a given tensor.
 *
 * This utility function inspects an existing tensor and extracts its device (e.g., CPU, CUDA)
 * and data type (e.g., `torch::kFloat32`, `torch::kInt64`). It then constructs and returns a
 * `torch::TensorOptions` object initialized with these properties. This is useful for creating
 * new tensors that should reside on the same device and have the same data type as a reference tensor,
 * without needing to manually query and specify these properties.
 *
 * @param x A pointer to the `torch::Tensor` whose properties (device and data type) will be used
 *          to configure the returned `TensorOptions`. The function expects `x` to point to a
 *          valid, initialized tensor object. The tensor pointed to by `x` is not modified.
 *          Behavior is undefined if `x` is a null pointer.
 *
 * @return torch::TensorOptions A `torch::TensorOptions` object configured with the same device
 *         and data type as the tensor pointed to by `x`. This options object can then be used,
 *         for example, with `torch::zeros(sizes, options)` or `torch::randn(sizes, options)`
 *         to create new tensors with matching properties.
 *
 * @throws May potentially throw exceptions if accessing properties of an invalid tensor state,
 *         although typically safe if `x` points to a valid tensor.
 */
torch::TensorOptions MakeOp(torch::Tensor* x);

/**
 * @brief Transfers a given PyTorch tensor to a specified computational device (e.g., CPU or a specific GPU).
 *
 * This function facilitates the movement of tensor data between different hardware devices
 * supported by PyTorch. It takes a pointer to an existing tensor and a string identifier
 * for the target device. It then creates and returns a *new* tensor containing the same data
 * as the input tensor, but allocated on the specified target device. The original tensor
 * pointed to by `inx` remains unchanged on its original device. This is analogous to the
 * `.to(device)` method in Python PyTorch.
 *
 * @param dev A `std::string` representing the target device. This string should follow PyTorch's
 *            device naming conventions. Common examples include:
 *            - `"cpu"`: For the system's main CPU.
 *            - `"cuda"`: Usually defaults to the primary CUDA-enabled GPU (equivalent to `"cuda:0"`).
 *            - `"cuda:0"`, `"cuda:1"`, etc.: For specific CUDA-enabled GPUs, indexed starting from 0.
 *            - `"mps"`: For Apple Metal Performance Shaders (on supported macOS versions).
 *            The availability of devices like `"cuda"` or `"mps"` depends on the PyTorch build configuration
 *            and the system's hardware capabilities. An invalid or unsupported device string will
 *            likely result in a runtime error.
 * @param inx A pointer to the source `torch::Tensor` that needs to be moved. The function
 *            accesses the data of the tensor pointed to by `inx` but does not modify the
 *            original tensor itself. The caller is responsible for ensuring that `inx` points
 *            to a valid, initialized tensor object. The function does not take ownership of
 *            the pointer or the tensor it points to. Behavior is undefined if `inx` is null.
 *
 * @return torch::Tensor A *new* `torch::Tensor` object residing on the device specified by the `dev`
 *         parameter. This new tensor holds a copy of the data from the tensor pointed to by `inx`.
 *         If the input tensor is already on the target device, this function still typically
 *         returns a new tensor (a copy), although PyTorch might have internal optimizations.
 *
 * @throws std::runtime_error (or similar PyTorch-specific exception) This function may throw an
 *         exception if the device string `dev` is malformed, refers to an unavailable device
 *         (e.g., requesting `"cuda:1"` when only one GPU exists or CUDA support is not compiled),
 *         or if memory allocation fails on the target device during the copy.
 *
 * @note Ensure that the PyTorch library is correctly initialized before calling this function,
 *       especially when targeting GPU devices. The validity and interpretation of the `dev`
 *       string are handled by the PyTorch C++ API (`torch::Device` constructor).
 */
torch::Tensor changedev(std::string dev, torch::Tensor* inx);

/**
 * @brief Attempts to change the device of a tensor potentially in-place or to a default device.
 *        (Note: Implementation details are missing, and the `void` return type is unusual for device changes).
 *
 * This function overload is intended to change the device of the tensor pointed to by `inpt`.
 * However, without an implementation, its exact behavior is unclear. Given the `void` return type
 * and the pointer argument, it might be intended to modify the tensor *in-place* (which is not
 * standard PyTorch behavior for `.to()`) or reassign the pointer `*inpt` to a new tensor on a
 * different (possibly default) device. For example, it could potentially implement `*inpt = inpt->to(default_device);`.
 *
 * @param inpt A pointer to the `torch::Tensor` whose device is intended to be changed.
 *             The function expects `inpt` to point to a valid tensor object. Depending on the
 *             (missing) implementation, the tensor object itself might be modified, or the pointer
 *             `*inpt` might be reassigned to point to a new tensor. Behavior is undefined if
 *             `inpt` is a null pointer.
 *
 * @return void This function does not return a value, suggesting a side effect on the input pointer
 *         or the object it points to.
 *
 * @warning The exact behavior of this function is unknown due to the missing implementation.
 *          Standard PyTorch device transfer operations (`.to()`) return a *new* tensor and do not
 *          modify the original tensor in-place, nor do they typically operate via non-const pointers
 *          with `void` return for this purpose. Use with caution or clarify its implementation.
 *          It might target a default device (e.g., default CUDA device if available, else CPU).
 *
 * @throws Potential exceptions depend heavily on the missing implementation. Could throw if
 *         device transfer fails (e.g., memory allocation, invalid default device).
 */
void changedev(torch::Tensor* inpt);

