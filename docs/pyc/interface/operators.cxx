/**
 * @file operators.cxx
 * @brief This file defines the interface for the operators module.
 */

#include <torch/torch.h>
#include <tuple>

namespace pyc {
namespace operators {

/**
 * @brief Calculates the dot product of two tensors.
 * @param v1 The first tensor.
 * @param v2 The second tensor.
 * @return The dot product of v1 and v2.
 */
torch::Tensor Dot(torch::Tensor v1, torch::Tensor v2);

/**
 * @brief Calculates the cosine of the angle between two tensors.
 * @param v1 The first tensor.
 * @param v2 The second tensor.
 * @return The cosine of the angle between v1 and v2.
 */
torch::Tensor CosTheta(torch::Tensor v1, torch::Tensor v2);

/**
 * @brief Calculates the sine of the angle between two tensors.
 * @param v1 The first tensor.
 * @param v2 The second tensor.
 * @return The sine of the angle between v1 and v2.
 */
torch::Tensor SinTheta(torch::Tensor v1, torch::Tensor v2);

/**
 * @brief Generates a rotation matrix around the X-axis.
 * @param angle The rotation angle in radians.
 * @return The rotation matrix.
 */
torch::Tensor Rx(torch::Tensor angle);

/**
 * @brief Generates a rotation matrix around the Y-axis.
 * @param angle The rotation angle in radians.
 * @return The rotation matrix.
 */
torch::Tensor Ry(torch::Tensor angle);

/**
 * @brief Generates a rotation matrix around the Z-axis.
 * @param angle The rotation angle in radians.
 * @return The rotation matrix.
 */
torch::Tensor Rz(torch::Tensor angle);

/**
 * @brief Applies a rotation and translation to a tensor.
 * @param pmc_b The translation tensor.
 * @param pmc_mu The rotation tensor.
 * @return The transformed tensor.
 */
torch::Tensor RT(torch::Tensor pmc_b, torch::Tensor pmc_mu);

/**
 * @brief Calculates the cofactors of a matrix.
 * @param matrix The input matrix.
 * @return The matrix of cofactors.
 */
torch::Tensor CoFactors(torch::Tensor matrix);

/**
 * @brief Calculates the determinant of a matrix.
 * @param matrix The input matrix.
 * @return The determinant of the matrix.
 */
torch::Tensor Determinant(torch::Tensor matrix);

/**
 * @brief Calculates the inverse of a matrix.
 * @param matrix The input matrix.
 * @return A tuple containing the inverse matrix and the determinant.
 */
std::tuple<torch::Tensor, torch::Tensor> Inverse(torch::Tensor matrix);

/**
 * @brief Calculates the cross product of two tensors.
 * @param mat1 The first tensor.
 * @param mat2 The second tensor.
 * @return The cross product of mat1 and mat2.
 */
torch::Tensor Cross(torch::Tensor mat1, torch::Tensor mat2);

/**
 * @brief Calculates the eigenvalues of a matrix.
 * @param matrix The input matrix.
 * @return A tuple containing the real and imaginary parts of the eigenvalues.
 */
std::tuple<torch::Tensor, torch::Tensor> Eigenvalue(torch::Tensor matrix);

} // namespace operators
} // namespace pyc
