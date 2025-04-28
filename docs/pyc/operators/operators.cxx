/**
 * @brief Computes the dot product of two tensors.
 * @param v1 First tensor pointer.
 * @param v2 Second tensor pointer.
 * @return Resulting tensor.
 */
torch::Tensor operators_::Dot(torch::Tensor* v1, torch::Tensor* v2);

/**
 * @brief Computes the cosine of the angle between two tensors.
 * @param v1 First tensor pointer.
 * @param v2 Second tensor pointer.
 * @return Resulting tensor.
 */
torch::Tensor operators_::CosTheta(torch::Tensor* v1, torch::Tensor* v2);

/**
 * @brief Computes the sine of the angle between two tensors.
 * @param v1 First tensor pointer.
 * @param v2 Second tensor pointer.
 * @return Resulting tensor.
 */
torch::Tensor operators_::SinTheta(torch::Tensor* v1, torch::Tensor* v2);

/**
 * @brief Builds a rotation matrix around the X axis.
 * @param angle Angle tensor pointer.
 * @return Rotation matrix.
 */
torch::Tensor operators_::Rx(torch::Tensor* angle);

/**
 * @brief Builds a rotation matrix around the Y axis.
 * @param angle Angle tensor pointer.
 * @return Rotation matrix.
 */
torch::Tensor operators_::Ry(torch::Tensor* angle);

/**
 * @brief Builds a rotation matrix around the Z axis.
 * @param angle Angle tensor pointer.
 * @return Rotation matrix.
 */
torch::Tensor operators_::Rz(torch::Tensor* angle);

/**
 * @brief Computes cofactor matrix.
 * @param matrix Matrix tensor pointer.
 * @return Cofactor matrix tensor.
 */
torch::Tensor operators_::CoFactors(torch::Tensor* matrix);

/**
 * @brief Computes the determinant of a matrix.
 * @param matrix Matrix tensor pointer.
 * @return Determinant tensor.
 */
torch::Tensor operators_::Determinant(torch::Tensor* matrix);

/**
 * @brief Computes the inverse of a matrix.
 * @param matrix Matrix tensor pointer.
 * @return Inverse matrix tensor.
 */
torch::Tensor operators_::Inverse(torch::Tensor* matrix);
