namespace operators_ {

/**
 * @brief Computes the dot product of two tensors.
 * @param v1 First input tensor pointer.
 * @param v2 Second input tensor pointer.
 * @return The resulting tensor.
 */
torch::Tensor Dot(torch::Tensor* v1, torch::Tensor* v2);

/**
 * @brief Computes the cross product of two tensors.
 * @param v1 First input tensor pointer.
 * @param v2 Second input tensor pointer.
 * @return The resulting tensor.
 */
torch::Tensor Cross(torch::Tensor* v1, torch::Tensor* v2);

/**
 * @brief Computes the cosine of the angle between two tensors.
 * @param v1 First input tensor pointer.
 * @param v2 Second input tensor pointer.
 * @return The resulting tensor.
 */
torch::Tensor CosTheta(torch::Tensor* v1, torch::Tensor* v2);

/**
 * @brief Computes the sine of the angle between two tensors.
 * @param v1 First input tensor pointer.
 * @param v2 Second input tensor pointer.
 * @return The resulting tensor.
 */
torch::Tensor SinTheta(torch::Tensor* v1, torch::Tensor* v2);

/**
 * @brief Constructs a rotation matrix around the x-axis.
 * @param angle Angle tensor pointer.
 * @return The resulting tensor.
 */
torch::Tensor Rx(torch::Tensor* angle);

/**
 * @brief Constructs a rotation matrix around the y-axis.
 * @param angle Angle tensor pointer.
 * @return The resulting tensor.
 */
torch::Tensor Ry(torch::Tensor* angle);

/**
 * @brief Constructs a rotation matrix around the z-axis.
 * @param angle Angle tensor pointer.
 * @return The resulting tensor.
 */
torch::Tensor Rz(torch::Tensor* angle);

/**
 * @brief Computes the cofactor matrix of a given matrix.
 * @param matrix Input matrix tensor pointer.
 * @return The resulting tensor.
 */
torch::Tensor CoFactors(torch::Tensor* matrix);

/**
 * @brief Computes the determinant of a given matrix.
 * @param matrix Input matrix tensor pointer.
 * @return The resulting tensor.
 */
torch::Tensor Determinant(torch::Tensor* matrix);

/**
 * @brief Computes the inverse of a given matrix.
 * @param matrix Input matrix tensor pointer.
 * @return The resulting tensor.
 */
torch::Tensor Inverse(torch::Tensor* matrix);

}  // namespace operators_
