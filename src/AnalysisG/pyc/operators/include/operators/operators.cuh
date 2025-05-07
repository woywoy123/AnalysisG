/**
 * @file operators.cuh
 * @brief Provides operator functions for mathematical operations in CUDA.
 */

#ifndef CU_OPERATORS_H
#define CU_OPERATORS_H

#include <torch/torch.h> ///< Includes PyTorch C++ headers for tensor operations.

/**
 * @brief Namespace for operator functions.
 */
namespace operators_ {

/**
 * @brief Computes the dot product of two vectors.
 *
 * @param v1 Pointer to the first input tensor.
 * @param v2 Pointer to the second input tensor.
 * @return A tensor containing the dot product result.
 */
torch::Tensor Dot(torch::Tensor* v1, torch::Tensor* v2);

/**
 * @brief Computes the cross product of two vectors.
 *
 * @param v1 Pointer to the first input tensor.
 * @param v2 Pointer to the second input tensor.
 * @return A tensor containing the cross product result.
 */
torch::Tensor Cross(torch::Tensor* v1, torch::Tensor* v2);

/**
 * @brief Computes the cosine of the angle between two vectors.
 *
 * @param v1 Pointer to the first input tensor.
 * @param v2 Pointer to the second input tensor.
 * @param lm Optional parameter for additional computation.
 * @return A tensor containing the cosine of the angle.
 */
torch::Tensor CosTheta(torch::Tensor* v1, torch::Tensor* v2, unsigned int lm = 0);

/**
 * @brief Computes the sine of the angle between two vectors.
 *
 * @param v1 Pointer to the first input tensor.
 * @param v2 Pointer to the second input tensor.
 * @param lm Optional parameter for additional computation.
 * @return A tensor containing the sine of the angle.
 */
torch::Tensor SinTheta(torch::Tensor* v1, torch::Tensor* v2, unsigned int lm = 0);

/**
 * @brief Computes the rotation matrix around the X-axis.
 *
 * @param angle Pointer to the input tensor representing the angle.
 * @return A tensor containing the rotation matrix.
 */
torch::Tensor Rx(torch::Tensor* angle);

/**
 * @brief Computes the rotation matrix around the Y-axis.
 *
 * @param angle Pointer to the input tensor representing the angle.
 * @return A tensor containing the rotation matrix.
 */
torch::Tensor Ry(torch::Tensor* angle);

/**
 * @brief Computes the rotation matrix around the Z-axis.
 *
 * @param angle Pointer to the input tensor representing the angle.
 * @return A tensor containing the rotation matrix.
 */
torch::Tensor Rz(torch::Tensor* angle);

/**
 * @brief Computes the rotation and translation matrix.
 *
 * @param pmu Pointer to the input tensor for translation.
 * @param phi Pointer to the input tensor for rotation around Z-axis.
 * @param theta Pointer to the input tensor for rotation around Y-axis.
 * @return A tensor containing the rotation and translation matrix.
 */
torch::Tensor RT(torch::Tensor* pmu, torch::Tensor* phi, torch::Tensor* theta);

/**
 * @brief Computes the cofactor matrix of a given matrix.
 *
 * @param matrix Pointer to the input tensor representing the matrix.
 * @return A tensor containing the cofactor matrix.
 */
torch::Tensor CoFactors(torch::Tensor* matrix);

/**
 * @brief Computes the determinant of a given matrix.
 *
 * @param matrix Pointer to the input tensor representing the matrix.
 * @return A tensor containing the determinant.
 */
torch::Tensor Determinant(torch::Tensor* matrix);

/**
 * @brief Computes the inverse of a given matrix.
 *
 * @param matrix Pointer to the input tensor representing the matrix.
 * @return A tuple containing the inverse matrix and additional information.
 */
std::tuple<torch::Tensor, torch::Tensor> Inverse(torch::Tensor* matrix);

/**
 * @brief Computes the eigenvalues of a given matrix.
 *
 * @param matrix Pointer to the input tensor representing the matrix.
 * @return A tuple containing the eigenvalues and additional information.
 */
std::tuple<torch::Tensor, torch::Tensor> Eigenvalue(torch::Tensor* matrix);

}

#endif
