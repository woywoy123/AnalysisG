#ifndef TRANSFORM_H
#define TRANSFORM_H
#include <torch/torch.h>

/**
 * @brief Namespace containing functions for kinematic transformations using torch::Tensor.
 *
 * This namespace provides utilities to convert between different representations
 * of particle kinematics, such as Cartesian (Px, Py, Pz, E) and
 * cylindrical (Pt, Eta, Phi, E) coordinates. All functions operate on
 * torch::Tensor objects, allowing for batch processing and GPU acceleration.
 * Input tensors are passed by pointer.
 */
namespace transform_ {
    /**
     * @brief Calculates the x-component of momentum (Px).
     * @param pt Pointer to a tensor containing transverse momentum values.
     * @param phi Pointer to a tensor containing azimuthal angle (phi) values in radians.
     * @return A tensor containing the calculated Px values.
     */
    torch::Tensor Px(torch::Tensor* pt, torch::Tensor* phi);

    /**
     * @brief Calculates the y-component of momentum (Py).
     * @param pt Pointer to a tensor containing transverse momentum values.
     * @param phi Pointer to a tensor containing azimuthal angle (phi) values in radians.
     * @return A tensor containing the calculated Py values.
     */
    torch::Tensor Py(torch::Tensor* pt, torch::Tensor* phi);

    /**
     * @brief Calculates the z-component of momentum (Pz).
     * @param pt Pointer to a tensor containing transverse momentum values.
     * @param eta Pointer to a tensor containing pseudorapidity (eta) values.
     * @return A tensor containing the calculated Pz values.
     */
    torch::Tensor Pz(torch::Tensor* pt, torch::Tensor* eta);

    /**
     * @brief Calculates Cartesian momentum components (Px, Py, Pz) from cylindrical coordinates.
     * @param pt Pointer to a tensor containing transverse momentum values.
     * @param eta Pointer to a tensor containing pseudorapidity (eta) values.
     * @param phi Pointer to a tensor containing azimuthal angle (phi) values in radians.
     * @return A tensor containing [Px, Py, Pz] column vectors.
     */
    torch::Tensor PxPyPz(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi);

    /**
     * @brief Calculates Cartesian 4-momentum (Px, Py, Pz, E) from cylindrical coordinates and energy.
     * @param pt Pointer to a tensor containing transverse momentum values.
     * @param eta Pointer to a tensor containing pseudorapidity (eta) values.
     * @param phi Pointer to a tensor containing azimuthal angle (phi) values in radians.
     * @param energy Pointer to a tensor containing energy values.
     * @return A tensor containing [Px, Py, Pz, E] column vectors.
     */
    torch::Tensor PxPyPzE(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi, torch::Tensor* energy);

    /**
     * @brief Extracts Cartesian momentum components (Px, Py, Pz) from a 4-momentum tensor.
     * Assumes the input tensor has shape (N, 4) where columns are [Px, Py, Pz, E] or similar.
     * @param pmu Pointer to a tensor containing 4-momentum vectors [Px, Py, Pz, E].
     * @return A tensor containing [Px, Py, Pz] column vectors.
     */
    torch::Tensor PxPyPz(torch::Tensor* pmu);

    /**
     * @brief Extracts Cartesian 4-momentum (Px, Py, Pz, E) from a 4-momentum tensor.
     * This function might simply return the input tensor if it's already in the desired format.
     * Assumes the input tensor has shape (N, 4) where columns are [Px, Py, Pz, E].
     * @param pmu Pointer to a tensor containing 4-momentum vectors [Px, Py, Pz, E].
     * @return A tensor containing [Px, Py, Pz, E] column vectors.
     */
    torch::Tensor PxPyPzE(torch::Tensor* pmu);

    /**
     * @brief Calculates the transverse momentum (Pt).
     * @param px Pointer to a tensor containing x-component of momentum values.
     * @param py Pointer to a tensor containing y-component of momentum values.
     * @return A tensor containing the calculated Pt values.
     */
    torch::Tensor Pt(torch::Tensor* px, torch::Tensor* py);

    /**
     * @brief Calculates pseudorapidity (Eta) from Pt and Pz.
     * @param pt Pointer to a tensor containing transverse momentum values.
     * @param pz Pointer to a tensor containing z-component of momentum values.
     * @return A tensor containing the calculated Eta values.
     * @note The function name `PtEta` might be misleading; it calculates Eta based on Pt and Pz.
     */
    torch::Tensor PtEta(torch::Tensor* pt, torch::Tensor* pz);

    /**
     * @brief Calculates the azimuthal angle (Phi) from a Cartesian momentum tensor.
     * Uses Px and Py components. Assumes the input tensor has shape (N, >=2)
     * where columns 0 and 1 are Px and Py respectively.
     * @param pmc Pointer to a tensor containing Cartesian momentum components (e.g., [Px, Py, Pz] or [Px, Py, Pz, E]).
     * @return A tensor containing the calculated Phi values in radians.
     */
    torch::Tensor Phi(torch::Tensor* pmc);

    /**
     * @brief Calculates the azimuthal angle (Phi).
     * @param px Pointer to a tensor containing x-component of momentum values.
     * @param py Pointer to a tensor containing y-component of momentum values.
     * @return A tensor containing the calculated Phi values in radians.
     */
    torch::Tensor Phi(torch::Tensor* px, torch::Tensor* py);

    /**
     * @brief Calculates the pseudorapidity (Eta) from a Cartesian momentum tensor.
     * Uses Px, Py, and Pz components. Assumes the input tensor has shape (N, >=3)
     * where columns 0, 1, and 2 are Px, Py, and Pz respectively.
     * @param pmc Pointer to a tensor containing Cartesian momentum components (e.g., [Px, Py, Pz] or [Px, Py, Pz, E]).
     * @return A tensor containing the calculated Eta values.
     */
    torch::Tensor Eta(torch::Tensor* pmc);

    /**
     * @brief Calculates the pseudorapidity (Eta).
     * @param px Pointer to a tensor containing x-component of momentum values.
     * @param py Pointer to a tensor containing y-component of momentum values.
     * @param pz Pointer to a tensor containing z-component of momentum values.
     * @return A tensor containing the calculated Eta values.
     */
    torch::Tensor Eta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

    /**
     * @brief Calculates cylindrical coordinates (Pt, Eta, Phi) from a Cartesian momentum tensor.
     * Assumes the input tensor has shape (N, >=3) where columns 0, 1, and 2 are Px, Py, and Pz.
     * @param pmc Pointer to a tensor containing Cartesian momentum components (e.g., [Px, Py, Pz] or [Px, Py, Pz, E]).
     * @return A tensor containing [Pt, Eta, Phi] column vectors.
     */
    torch::Tensor PtEtaPhi(torch::Tensor* pmc);

    /**
     * @brief Calculates cylindrical coordinates (Pt, Eta, Phi) and extracts energy (E) from a Cartesian 4-momentum tensor.
     * Assumes the input tensor has shape (N, 4) where columns are [Px, Py, Pz, E].
     * @param pmc Pointer to a tensor containing Cartesian 4-momentum vectors [Px, Py, Pz, E].
     * @return A tensor containing [Pt, Eta, Phi, E] column vectors.
     */
    torch::Tensor PtEtaPhiE(torch::Tensor* pmc);

    /**
     * @brief Calculates cylindrical coordinates (Pt, Eta, Phi) from Cartesian momentum components.
     * @param px Pointer to a tensor containing x-component of momentum values.
     * @param py Pointer to a tensor containing y-component of momentum values.
     * @param pz Pointer to a tensor containing z-component of momentum values.
     * @return A tensor containing [Pt, Eta, Phi] column vectors.
     */
    torch::Tensor PtEtaPhi(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

    /**
     * @brief Calculates cylindrical coordinates (Pt, Eta, Phi) and includes energy (E) from Cartesian components.
     * @param px Pointer to a tensor containing x-component of momentum values.
     * @param py Pointer to a tensor containing y-component of momentum values.
     * @param pz Pointer to a tensor containing z-component of momentum values.
     * @param e Pointer to a tensor containing energy values.
     * @return A tensor containing [Pt, Eta, Phi, E] column vectors.
     */
    torch::Tensor PtEtaPhiE(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);
}

#endif
