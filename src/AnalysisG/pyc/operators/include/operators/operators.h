/**
 * @file operators.h
 * @brief Provides operator functions for mathematical operations in C++.
 */

#ifndef OPERATORS_H
#define OPERATORS_H

#include <torch/torch.h> ///< Includes PyTorch C++ headers for tensor operations.

/**
 * @brief Namespace for operator functions.
 */
namespace operators_ {

/**
 * @brief Computes the dot product of two vectors.
 *
 * @param v1 Input tensor representing the first vector.
 * @param v2 Input tensor representing the second vector.
 * @return A tensor containing the dot product result.
 */
torch::Tensor Dot(torch::Tensor* v1, torch::Tensor* v2);

/**
 * @brief Computes the cross product of two vectors.
 *
 * @param v1 Input tensor representing the first vector.
 * @param v2 Input tensor representing the second vector.
 * @return A tensor containing the cross product result.
 */
torch::Tensor Cross(torch::Tensor* v1, torch::Tensor* v2);

/**
 * @brief Computes the cosine of the angle between two vectors.
 *
 * @param v1 Input tensor representing the first vector.
 * @param v2 Input tensor representing the second vector.
 * @return A tensor containing the cosine of the angle.
 */
torch::Tensor CosTheta(torch::Tensor* v1, torch::Tensor* v2);

/**
 * @brief Computes the sine of the angle between two vectors.
 *
 * @param v1 Input tensor representing the first vector.
 * @param v2 Input tensor representing the second vector.
 * @return A tensor containing the sine of the angle.
 */
torch::Tensor SinTheta(torch::Tensor* v1, torch::Tensor* v2);

/**
 * @brief Computes the rotation matrix around the X-axis.
 *
 * @param angle Input tensor representing the rotation angle.
 * @return A tensor containing the rotation matrix.
 */
torch::Tensor Rx(torch::Tensor* angle);

/**
 * @brief Computes the rotation matrix around the Y-axis.
 *
 * @param angle Input tensor representing the rotation angle.
 * @return A tensor containing the rotation matrix.
 */
torch::Tensor Ry(torch::Tensor* angle);

/**
 * @brief Computes the rotation matrix around the Z-axis.
 *
 * @param angle Input tensor representing the rotation angle.
 * @return A tensor containing the rotation matrix.
 */
torch::Tensor Rz(torch::Tensor* angle);

/**
 * @brief Computes the cofactor matrix of a given matrix.
 *
 * @param matrix Input tensor representing the matrix.
 * @return A tensor containing the cofactor matrix.
 */
torch::Tensor CoFactors(torch::Tensor* matrix);

/**
 * @brief Computes the determinant of a matrix.
 *
 * @param matrix Input tensor representing the matrix.
 * @return A tensor containing the determinant value.
 */
torch::Tensor Determinant(torch::Tensor* matrix);

/**
 * @brief Computes the inverse of a matrix.
 *
 * @param matrix Input tensor representing the matrix.
 * @return A tensor containing the inverse matrix.
 */
torch::Tensor Inverse(torch::Tensor* matrix);

} // namespace operators_

#endif // OPERATORS_H
