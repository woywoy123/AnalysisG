/**
 * @file device.cuh
 * @brief Defines the core CUDA device-side structures and function prototypes for the NuSol algorithm.
 * @details This header file declares the `nusol` structure, which encapsulates all necessary parameters
 * and intermediate values for the neutrino momentum reconstruction algorithm (NuSol) executed on the GPU.
 * It also declares several `__device__ __forceinline__` functions that implement the mathematical
 * steps of the NuSol calculation within CUDA kernels. These functions are designed to be called
 * from CUDA kernels operating on multiple events in parallel.
 */

#ifndef CU_NUSOL_DEVICE_H
#define CU_NUSOL_DEVICE_H

// Include necessary utility headers, specifically for atomic operations which might be used
// in conjunction with the NuSol results aggregation or other parallel operations.
#include <utils/atomic.cuh>

/**
 * @brief Encapsulates parameters and state for a single NuSol calculation instance on the device.
 * @details This structure holds all input parameters (like particle masses, momenta, detector information)
 * and intermediate calculated values required by the NuSol algorithm steps. An instance of this
 * structure typically corresponds to one physics event being processed on the GPU. The members
 * are carefully laid out to potentially optimize memory access patterns on the device.
 */
struct nusol {
    /** @brief Cosine of the rotation angle used in the transformation to the t-tbar frame or a similar reference frame. */
    double cos;
    /** @brief Sine of the rotation angle used in the transformation. */
    double sin;
    /** @brief Transformed x-coordinate (or related parameter) often representing a component of momentum or position. */
    double x0;
    /** @brief Derivative or related quantity associated with x0, potentially representing a momentum component. */
    double x0p;
    /** @brief Scaling factor or component related to the x-dimension, possibly related to detector resolution or particle kinematics. */
    double sx;
    /** @brief Scaling factor or component related to the y-dimension. */
    double sy;
    /** @brief Parameter 'w', often related to the W boson mass constraint or a similar kinematic variable. */
    double w;
    /** @brief Parameter related to 'w', potentially an intermediate calculation or a modified constraint value. */
    double w_;
    /** @brief Transformed x-coordinate (or related parameter) representing a different point or component than x0. */
    double x1;
    /** @brief Transformed y-coordinate (or related parameter) associated with x1. */
    double y1;
    /** @brief Transformed z-coordinate or a parameter related to the longitudinal component. */
    double z;
    /** @brief Intermediate calculated value, possibly related to squared quantities or offsets (o2 might suggest 'offset squared' or similar). */
    double o2;
    /** @brief Small value used for numerical stability or tolerance checks, likely related to squared quantities (eps2 might suggest 'epsilon squared'). */
    double eps2;

    /** @brief Array storing the masses of the particles involved, indexed [0] for lepton, [1] for b-quark. Used for kinematic calculations. */
    double pmass[2];
    /** @brief Array storing the beta values (velocity/c) of the particles, indexed [0] for lepton, [1] for b-quark. */
    double betas[2];
    /** @brief Four-momentum (E, px, py, pz) of the b-quark involved in the decay. */
    double pmu_b[4];
    /** @brief Four-momentum (E, px, py, pz) of the lepton involved in the decay. */
    double pmu_l[4];
    /** @brief Array storing relevant masses, potentially including W boson mass, top quark mass, etc. Indexing depends on the specific NuSol implementation context. */
    double masses[3];
    /** @brief Boolean flag indicating whether the calculated neutrino solution(s) for this event satisfy the required physical or geometric criteria. Set to true if valid solutions are found. */
    bool passed;

    /**
     * @brief Default constructor.
     * @details Initializes a nusol struct with default values. Ensures that the object is created
     * in a valid, albeit potentially uninitialized, state before parameters are explicitly set.
     * The `= default` specification requests the compiler to generate a default constructor.
     */
    nusol() = default;
};

/**
 * @brief Executes the main NuSol algorithm steps on the device for a single event.
 * @details This function orchestrates the sequence of calculations required by the NuSol algorithm,
 * utilizing other helper device functions (_krot, _amu, _abq, _htilde, etc.). It takes the input
 * parameters stored within the `nusol` struct, performs the intermediate calculations, solves
 * for the neutrino momentum components, and potentially updates the `passed` flag based on the
 * validity and number of solutions found. Being `__device__ __forceinline__`, it's intended to be
 * directly inlined into the calling kernel for performance.
 * @param sl Pointer to the `nusol` struct instance for the current event being processed by the thread.
 *           The function reads input parameters from and writes results (like intermediate values
 *           or the final `passed` status) back to this struct.
 * @note This function encapsulates the core logic of the NuSol solver on the GPU.
 */
__device__ __forceinline__ void _makeNuSol(nusol* sl);

/**
 * @brief Performs a rotation calculation, likely transforming momentum components.
 * @details This function calculates a value based on a rotation transformation defined by the
 * `cos` and `sin` members of the `nusol` struct. The indices `_iky` and `_ikz` likely select
 * specific components or elements involved in the rotation, possibly related to momentum components
 * in different coordinate systems (e.g., ky, kz components). The exact transformation depends
 * on the specific formulation of NuSol being implemented.
 * @param sl Pointer to the `nusol` struct containing rotation parameters (`cos`, `sin`) and potentially
 *           other relevant data needed for the calculation.
 * @param _iky An index, potentially representing the y-component or a related index (0, 1, or 2).
 * @param _ikz An index, potentially representing the z-component or a related index (0, 1, or 2).
 * @return The result of the rotation calculation, typically a transformed component or a dot product.
 * @note Marked `__device__ __forceinline__` for performance within CUDA kernels.
 */
__device__ __forceinline__ double _krot(nusol* sl, const unsigned int _iky, const unsigned int _ikz);

/**
 * @brief Calculates the 'amu' intermediate value for the NuSol algorithm.
 * @details This function computes a specific intermediate quantity, denoted 'amu', which is part of
 * the system of equations solved by NuSol. It likely involves kinematic variables of the lepton,
 * potentially related to its momentum components after rotation, as suggested by the dependency
 * on `_iky` and `_ikz`. The exact formula depends on the NuSol derivation.
 * @param sl Pointer to the `nusol` struct containing necessary input parameters (e.g., lepton momentum, masses).
 * @param _iky An index, potentially selecting a specific component or term (0, 1, or 2).
 * @param _ikz An index, potentially selecting a specific component or term (0, 1, or 2).
 * @return The calculated 'amu' value.
 * @note Marked `__device__ __forceinline__` for performance within CUDA kernels.
 */
__device__ __forceinline__ double _amu(nusol* sl, const unsigned int _iky, const unsigned int _ikz);

/**
 * @brief Calculates the 'abq' intermediate value for the NuSol algorithm.
 * @details This function computes another specific intermediate quantity, denoted 'abq', required
 * for the NuSol solution. It likely involves kinematic variables of the b-quark, potentially
 * related to its momentum components after rotation, indicated by the `_iky` and `_ikz` parameters.
 * The exact formula depends on the NuSol derivation.
 * @param sl Pointer to the `nusol` struct containing necessary input parameters (e.g., b-quark momentum, masses).
 * @param _iky An index, potentially selecting a specific component or term (0, 1, or 2).
 * @param _ikz An index, potentially selecting a specific component or term (0, 1, or 2).
 * @return The calculated 'abq' value.
 * @note Marked `__device__ __forceinline__` for performance within CUDA kernels.
 */
__device__ __forceinline__ double _abq(nusol* sl, const unsigned int _iky, const unsigned int _ikz);

/**
 * @brief Calculates the 'Htilde' intermediate value for the NuSol algorithm.
 * @details This function computes the 'Htilde' quantity, another intermediate value crucial for
 * solving the NuSol equations. Its calculation typically involves combinations of lepton and
 * b-quark kinematics, masses, and potentially the W mass constraint. The `_iky` and `_ikz` indices
 * likely select specific terms or components contributing to Htilde.
 * @param sl Pointer to the `nusol` struct containing necessary input parameters and potentially
 *           previously calculated intermediate values.
 * @param _iky An index, potentially selecting a specific component or term (0, 1, or 2).
 * @param _ikz An index, potentially selecting a specific component or term (0, 1, or 2).
 * @return The calculated 'Htilde' value.
 * @note Marked `__device__ __forceinline__` for performance within CUDA kernels.
 */
__device__ __forceinline__ double _htilde(nusol* sl, const unsigned int _iky, const unsigned int _ikz);

/**
 * @brief Performs a calculation specific to 'case 1' of the NuSol solution derivation.
 * @details The NuSol algorithm often involves analyzing different geometric or algebraic cases.
 * This function implements the calculation specific to one such case, referred to as 'case 1'.
 * It likely operates on elements of a matrix `G` (possibly related to a metric tensor or a
 * coefficient matrix) using indices `_idy` and `_idz` to select specific elements or components.
 * @param G A 3x3 matrix (passed as a C-style array) containing coefficients or geometric factors relevant to case 1.
 * @param _idy Index (0, 1, or 2) specifying the row or component for the calculation.
 * @param _idz Index (0, 1, or 2) specifying the column or component for the calculation.
 * @return The result of the calculation specific to case 1.
 * @note Marked `__device__ __forceinline__` for performance. The exact nature of 'case 1' depends on the NuSol paper/implementation.
 */
__device__ __forceinline__ double _case1(double G[3][3], const unsigned int _idy, const unsigned int _idz);

/**
 * @brief Performs a calculation specific to 'case 2' of the NuSol solution derivation.
 * @details Similar to `_case1`, this function implements the calculation for 'case 2' of the NuSol
 * algorithm. It operates on the matrix `G` using indices `_idy` and `_idz`. The `swpXY` flag
 * suggests that this case might involve a symmetry or transformation where the roles of X and Y
 * coordinates (or related parameters) are potentially swapped or treated differently compared to case 1.
 * @param G A 3x3 matrix (passed as a C-style array) containing coefficients or geometric factors relevant to case 2.
 * @param _idy Index (0, 1, or 2) specifying the row or component.
 * @param _idz Index (0, 1, or 2) specifying the column or component.
 * @param swpXY Boolean flag indicating whether a swap or modification related to X/Y components should be applied.
 * @return The result of the calculation specific to case 2.
 * @note Marked `__device__ __forceinline__` for performance.
 */
__device__ __forceinline__ double _case2(double G[3][3], const unsigned int _idy, const unsigned int _idz, bool swpXY);

/**
 * @brief Calculates a component related to the lepton ellipse/ellipsoid intersection ('leqnulls').
 * @details This function likely calculates a term related to finding the intersection points in the
 * NuSol algorithm, specifically associated with the lepton constraints ('lepton quadratic nulls' or similar).
 * It operates on cofactor matrices (`coF`) and quadratic form matrices (`Q`), using indices `_idy`
 * and `_idz` to select specific matrix elements or components involved in the intersection calculation.
 * @param coF A 3x3 cofactor matrix derived from the system's equations.
 * @param Q A 3x3 matrix representing a quadratic form, likely related to the lepton momentum constraints.
 * @param _idy Index (0, 1, or 2) specifying the row or component.
 * @param _idz Index (0, 1, or 2) specifying the column or component.
 * @return The calculated value related to the lepton constraint intersection.
 * @note Marked `__device__ __forceinline__` for performance. 'leqnulls' likely refers to roots or nulls of an equation.
 */
__device__ __forceinline__ double _leqnulls(double coF[3][3], double Q[3][3], const unsigned int _idy, const unsigned int _idz);

/**
 * @brief Calculates a component related to the overall geometric intersection ('gnulls').
 * @details Similar to `_leqnulls`, this function calculates a term related to the geometric solution
 * of the NuSol system ('geometric nulls' or similar). It likely combines constraints or represents
 * coefficients in the final polynomial equation whose roots yield the neutrino momentum solutions.
 * It uses cofactor (`coF`) and quadratic form (`Q`) matrices with indices `_idy` and `_idz`.
 * @param coF A 3x3 cofactor matrix.
 * @param Q A 3x3 matrix representing a quadratic form, possibly related to the combined system constraints.
 * @param _idy Index (0, 1, or 2) specifying the row or component.
 * @param _idz Index (0, 1, or 2) specifying the column or component.
 * @return The calculated value related to the overall geometric intersection or equation coefficients.
 * @note Marked `__device__ __forceinline__` for performance.
 */
__device__ __forceinline__ double _gnulls(double coF[3][3], double Q[3][3], const unsigned int _idy, const unsigned int _idz);

#endif // CU_NUSOL_DEVICE_H
