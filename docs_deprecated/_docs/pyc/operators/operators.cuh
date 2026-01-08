namespace operators_ {
    /**
     * @brief Computes the dot product of two tensors.
     */
    torch::Tensor Dot(torch::Tensor* v1, torch::Tensor* v2);

    /**
     * @brief Computes the cross product of two tensors.
     */
    torch::Tensor Cross(torch::Tensor* v1, torch::Tensor* v2);

    /**
     * @brief Computes the cosine of the angle between two tensors.
     */
    torch::Tensor CosTheta(torch::Tensor* v1, torch::Tensor* v2, unsigned int lm = 0);

    /**
     * @brief Computes the sine of the angle between two tensors.
     */
    torch::Tensor SinTheta(torch::Tensor* v1, torch::Tensor* v2, unsigned int lm = 0);

    /**
     * @brief Creates a rotation around the x-axis.
     */
    torch::Tensor Rx(torch::Tensor* angle);

    /**
     * @brief Creates a rotation around the y-axis.
     */
    torch::Tensor Ry(torch::Tensor* angle);

    /**
     * @brief Creates a rotation around the z-axis.
     */
    torch::Tensor Rz(torch::Tensor* angle);

    /**
     * @brief Applies a rotation transform using PMU, phi, and theta.
     */
    torch::Tensor RT(torch::Tensor* pmu, torch::Tensor* phi, torch::Tensor* theta);

    /**
     * @brief Computes cofactor matrix.
     */
    torch::Tensor CoFactors(torch::Tensor* matrix);

    /**
     * @brief Computes determinant of the matrix.
     */
    torch::Tensor Determinant(torch::Tensor* matrix);

    /**
     * @brief Computes the inverse of the matrix and its determinant.
     */
    std::tuple<torch::Tensor, torch::Tensor> Inverse(torch::Tensor* matrix);

    /**
     * @brief Computes the eigenvalues and eigenvectors of the matrix.
     */
    std::tuple<torch::Tensor, torch::Tensor> Eigenvalue(torch::Tensor* matrix);
}
