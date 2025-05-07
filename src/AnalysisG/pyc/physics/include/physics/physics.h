/**
 * @file physics.h
 * @brief Provides physics-related functions for C++ operations.
 */

#ifndef PHYSICS_H
#define PHYSICS_H

#include <torch/torch.h> ///< Includes PyTorch C++ headers for tensor operations.

/**
 * @brief Namespace for physics-related functions.
 */
namespace physics_ {

/**
 * @brief Computes the square of the momentum magnitude.
 *
 * @param px Input tensor for the x-component of momentum.
 * @param py Input tensor for the y-component of momentum.
 * @param pz Input tensor for the z-component of momentum.
 * @return A tensor containing the square of the momentum magnitude.
 */
torch::Tensor P2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

/**
 * @brief Computes the square of the momentum magnitude from a momentum tensor.
 *
 * @param pmc Input tensor for the momentum components.
 * @return A tensor containing the square of the momentum magnitude.
 */
torch::Tensor P2(torch::Tensor* pmc);

/**
 * @brief Computes the momentum magnitude.
 *
 * @param px Input tensor for the x-component of momentum.
 * @param py Input tensor for the y-component of momentum.
 * @param pz Input tensor for the z-component of momentum.
 * @return A tensor containing the momentum magnitude.
 */
torch::Tensor P(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

/**
 * @brief Computes the momentum magnitude from a momentum tensor.
 *
 * @param pmc Input tensor for the momentum components.
 * @return A tensor containing the momentum magnitude.
 */
torch::Tensor P(torch::Tensor* pmc);

/**
 * @brief Computes the square of the beta factor.
 *
 * @param px Input tensor for the x-component of momentum.
 * @param py Input tensor for the y-component of momentum.
 * @param pz Input tensor for the z-component of momentum.
 * @param e Input tensor for the energy of the system.
 * @return A tensor containing the square of the beta factor.
 */
torch::Tensor Beta2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes the square of the beta factor from a momentum tensor.
 *
 * @param pmc Input tensor for the momentum components.
 * @return A tensor containing the square of the beta factor.
 */
torch::Tensor Beta2(torch::Tensor* pmc);

/**
 * @brief Computes the beta factor.
 *
 * @param px Input tensor for the x-component of momentum.
 * @param py Input tensor for the y-component of momentum.
 * @param pz Input tensor for the z-component of momentum.
 * @param e Input tensor for the energy of the system.
 * @return A tensor containing the beta factor.
 */
torch::Tensor Beta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes the beta factor from a momentum tensor.
 *
 * @param pmc Input tensor for the momentum components.
 * @return A tensor containing the beta factor.
 */
torch::Tensor Beta(torch::Tensor* pmc);

/**
 * @brief Computes the square of the invariant mass.
 *
 * @param px Input tensor for the x-component of momentum.
 * @param py Input tensor for the y-component of momentum.
 * @param pz Input tensor for the z-component of momentum.
 * @param e Input tensor for the energy of the system.
 * @return A tensor containing the square of the invariant mass.
 */
torch::Tensor M2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes the square of the invariant mass from a momentum tensor.
 *
 * @param pmc Input tensor for the momentum components.
 * @return A tensor containing the square of the invariant mass.
 */
torch::Tensor M2(torch::Tensor* pmc);

/**
 * @brief Computes the invariant mass.
 *
 * @param px Input tensor for the x-component of momentum.
 * @param py Input tensor for the y-component of momentum.
 * @param pz Input tensor for the z-component of momentum.
 * @param e Input tensor for the energy of the system.
 * @return A tensor containing the invariant mass.
 */
torch::Tensor M(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes the invariant mass from a momentum tensor.
 *
 * @param pmc Input tensor for the momentum components.
 * @return A tensor containing the invariant mass.
 */
torch::Tensor M(torch::Tensor* pmc);

/**
 * @brief Computes the square of the transverse mass.
 *
 * @param pz Input tensor for the z-component of momentum.
 * @param e Input tensor for the energy of the system.
 * @return A tensor containing the square of the transverse mass.
 */
torch::Tensor Mt2(torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes the square of the transverse mass from a momentum tensor.
 *
 * @param pmc Input tensor for the momentum components.
 * @return A tensor containing the square of the transverse mass.
 */
torch::Tensor Mt2(torch::Tensor* pmc);

/**
 * @brief Computes the transverse mass.
 *
 * @param pz Input tensor for the z-component of momentum.
 * @param e Input tensor for the energy of the system.
 * @return A tensor containing the transverse mass.
 */
torch::Tensor Mt(torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes the transverse mass from a momentum tensor.
 *
 * @param pmc Input tensor for the momentum components.
 * @return A tensor containing the transverse mass.
 */
torch::Tensor Mt(torch::Tensor* pmc);

/**
 * @brief Computes the polar angle theta.
 *
 * @param pmc Input tensor for the momentum components.
 * @return A tensor containing the polar angle theta.
 */
torch::Tensor Theta(torch::Tensor* pmc);

/**
 * @brief Computes the polar angle theta.
 *
 * @param px Input tensor for the x-component of momentum.
 * @param py Input tensor for the y-component of momentum.
 * @param pz Input tensor for the z-component of momentum.
 * @return A tensor containing the polar angle theta.
 */
torch::Tensor Theta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

/**
 * @brief Computes the delta R separation between two particles.
 *
 * @param pmu1 Input tensor for the first particle's momentum.
 * @param pmu2 Input tensor for the second particle's momentum.
 * @return A tensor containing the delta R separation.
 */
torch::Tensor DeltaR(torch::Tensor* pmu1, torch::Tensor* pmu2);

/**
 * @brief Computes the delta R separation between two particles.
 *
 * @param eta1 Input tensor for the pseudorapidity of the first particle.
 * @param eta2 Input tensor for the pseudorapidity of the second particle.
 * @param phi1 Input tensor for the azimuthal angle of the first particle.
 * @param phi2 Input tensor for the azimuthal angle of the second particle.
 * @return A tensor containing the delta R separation.
 */
torch::Tensor DeltaR(torch::Tensor* eta1, torch::Tensor* eta2, torch::Tensor* phi1, torch::Tensor* phi2);

}

#endif
