/**
 * @file pyc_cuda_kernels.h
 * @brief Documentation for CUDA kernels in the pyc module.
 *
 * @mainpage pyc CUDA Kernels Documentation
 *
 * @section overview Overview
 * This file documents the CUDA kernels implemented in the `pyc` module. These kernels are designed for high-performance computations in High Energy Physics (HEP) and are optimized for GPU execution. The kernels are implemented in `.cu` and `.cuh` files located in the `src/AnalysisG/pyc` directory.
 *
 * @section kernels CUDA Kernels
 *
 * @subsection utils_kernels Utils Kernels
 * - `__device__ float pyc::utils::dot_product(const float* vec1, const float* vec2, int size)`
 *   - Computes the dot product of two vectors on the GPU.
 *   - @param vec1 Pointer to the first vector.
 *   - @param vec2 Pointer to the second vector.
 *   - @param size The size of the vectors.
 *   - @return The dot product of the two vectors.
 *
 * - `__global__ void pyc::utils::normalize_vector(float* vec, int size)`
 *   - Normalizes a vector on the GPU.
 *   - @param vec Pointer to the vector to be normalized.
 *   - @param size The size of the vector.
 *
 * @subsection operators_kernels Operators Kernels
 * - `__global__ void pyc::operators::matrix_multiply(const float* mat1, const float* mat2, float* result, int rows, int cols, int common_dim)`
 *   - Performs matrix multiplication on the GPU.
 *   - @param mat1 Pointer to the first matrix.
 *   - @param mat2 Pointer to the second matrix.
 *   - @param result Pointer to the result matrix.
 *   - @param rows Number of rows in the result matrix.
 *   - @param cols Number of columns in the result matrix.
 *   - @param common_dim The common dimension of the input matrices.
 *
 * - `__global__ void pyc::operators::vector_add(const float* vec1, const float* vec2, float* result, int size)`
 *   - Adds two vectors element-wise on the GPU.
 *   - @param vec1 Pointer to the first vector.
 *   - @param vec2 Pointer to the second vector.
 *   - @param result Pointer to the result vector.
 *   - @param size The size of the vectors.
 *
 * @subsection nusol_kernels NuSol Kernels
 * - `__global__ void pyc::nusol::reconstruct_neutrino(const float* b_quark, const float* lepton, const float* met, float* neutrino, int size)`
 *   - Reconstructs the neutrino momentum on the GPU.
 *   - @param b_quark Pointer to the b-quark momentum.
 *   - @param lepton Pointer to the lepton momentum.
 *   - @param met Pointer to the missing transverse energy (MET).
 *   - @param neutrino Pointer to the reconstructed neutrino momentum.
 *   - @param size The size of the input arrays.
 *
 * - `__global__ void pyc::nusol::reconstruct_double_neutrino(const float* b_quark1, const float* b_quark2, const float* lepton1, const float* lepton2, const float* met, float* neutrinos, int size)`
 *   - Reconstructs the momenta of two neutrinos on the GPU.
 *   - @param b_quark1 Pointer to the first b-quark momentum.
 *   - @param b_quark2 Pointer to the second b-quark momentum.
 *   - @param lepton1 Pointer to the first lepton momentum.
 *   - @param lepton2 Pointer to the second lepton momentum.
 *   - @param met Pointer to the missing transverse energy (MET).
 *   - @param neutrinos Pointer to the reconstructed neutrino momenta.
 *   - @param size The size of the input arrays.
 *
 * @section usage Usage
 * These CUDA kernels are invoked from the C++ source code in the `pyc` module. They are designed to be called from host code using appropriate CUDA kernel launch configurations.
 */