/**
 * @brief Computes the dot product of two tensors.
 * @param v1 Pointer to the first input tensor.
 * @param v2 Pointer to the second input tensor.
 * @return Resulting dot product tensor.
 */
torch::Tensor operators_::Dot(torch::Tensor* v1, torch::Tensor* v2);

/**
 * @brief Computes the cross product of two tensors.
 * @param v1 Pointer to the first input tensor.
 * @param v2 Pointer to the second input tensor.
 * @return Resulting cross product tensor.
 */
torch::Tensor operators_::Cross(torch::Tensor* v1, torch::Tensor* v2);

/**
 * @brief Computes the cosine of the angle between two tensors.
 * @param v1 Pointer to the first input tensor.
 * @param v2 Pointer to the second input tensor.
 * @param lm Optional dimension limit.
 * @return Resulting tensor of cosine values.
 */
torch::Tensor operators_::CosTheta(torch::Tensor* v1, torch::Tensor* v2, unsigned int lm);

/**
 * @brief Computes the sine of the angle between two tensors.
 * @param v1 Pointer to the first input tensor.
 * @param v2 Pointer to the second input tensor.
 * @param lm Optional dimension limit.
 * @return Resulting tensor of sine values.
 */
torch::Tensor operators_::SinTheta(torch::Tensor* v1, torch::Tensor* v2, unsigned int lm);

/**
 * @brief Constructs a rotation matrix around the x-axis.
 * @param angle Pointer to the tensor of rotation angles.
 * @return Tensor of rotation matrices.
 */
torch::Tensor operators_::Rx(torch::Tensor* angle);

/**
 * @brief Constructs a rotation matrix around the y-axis.
 * @param angle Pointer to the tensor of rotation angles.
 * @return Tensor of rotation matrices.
 */
torch::Tensor operators_::Ry(torch::Tensor* angle);

/**
 * @brief Constructs a rotation matrix around the z-axis.
 * @param angle Pointer to the tensor of rotation angles.
 * @return Tensor of rotation matrices.
 */
torch::Tensor operators_::Rz(torch::Tensor* angle);

/**
 * @brief Builds a rotation matrix from momentum, azimuth, and polar angles.
 * @param pmc Pointer to the momentum tensor.
 * @param phi Pointer to the azimuth angle tensor.
 * @param theta Pointer to the polar angle tensor.
 * @return Tensor of rotation matrices.
 */
torch::Tensor operators_::RT(torch::Tensor* pmc, torch::Tensor* phi, torch::Tensor* theta);

/**
 * @brief Computes cofactor matrices of a batch of 3x3 matrices.
 * @param matrix Pointer to the input batch of matrices.
 * @return Tensor of cofactor matrices.
 */
torch::Tensor operators_::CoFactors(torch::Tensor* matrix);

/**
 * @brief Computes determinants of a batch of 3x3 matrices.
 * @param matrix Pointer to the input batch of matrices.
 * @return Tensor of determinant values.
 */
torch::Tensor operators_::Determinant(torch::Tensor* matrix);

/**
 * @brief Computes the inverse of a batch of 3x3 matrices and their determinants.
 * @param matrix Pointer to the input batch of matrices.
 * @return Tuple containing the inverse matrices and determinant values.
 */
std::tuple<torch::Tensor, torch::Tensor> operators_::Inverse(torch::Tensor* matrix);

/**
 * @brief Computes the eigenvalues of a batch of 3x3 matrices (real and imaginary parts).
 * @param matrix Pointer to the input batch of matrices.
 * @return Tuple containing real and imaginary eigenvalues.
 */
std::tuple<torch::Tensor, torch::Tensor> operators_::Eigenvalue(torch::Tensor* matrix);
