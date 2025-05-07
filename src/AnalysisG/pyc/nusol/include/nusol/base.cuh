/**
 * @file base.cuh
 * @brief Provides base definitions for the nusol module.
 */

#ifndef CU_NUSOL_BASE_H
#define CU_NUSOL_BASE_H

#include <map>
#include <string>
#include <torch/torch.h> ///< Includes PyTorch C++ headers for tensor operations.
#include <utils/atomic.cuh>
#include <utils/utils.cuh> ///< Includes utility functions for CUDA operations.

/**
 * @brief Namespace for neutrino solution (nusol) operations.
 */
namespace nusol_ {

/**
 * @brief Provides debugging information for the base module.
 *
 * @param pmc_b Pointer to the tensor containing b-quark momenta.
 * @param pmc_mu Pointer to the tensor containing muon momenta.
 * @param masses Pointer to the tensor containing mass values.
 * @return A map of strings to tensors containing debugging information.
 */
std::map<std::string, torch::Tensor> BaseDebug(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses); 

/**
 * @brief Computes the base matrix for the nusol module.
 *
 * @param pmc_b Pointer to the tensor containing b-quark momenta.
 * @param pmc_mu Pointer to the tensor containing muon momenta.
 * @param masses Pointer to the tensor containing mass values.
 * @return A map of strings to tensors containing the base matrix.
 */
std::map<std::string, torch::Tensor> BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses); 

/**
 * @brief Computes the base matrix for the nusol module with specific mass values.
 *
 * @param pmc_b Pointer to the tensor containing b-quark momenta.
 * @param pmc_mu Pointer to the tensor containing muon momenta.
 * @param mT Top quark mass.
 * @param mW W boson mass.
 * @param mN Neutrino mass.
 * @return A map of strings to tensors containing the base matrix.
 */
std::map<std::string, torch::Tensor> BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, double mT, double mW, double mN); 

/**
 * @brief Finds the intersection of two tensors.
 *
 * @param A Pointer to the first tensor.
 * @param B Pointer to the second tensor.
 * @param nulls Null value to use for intersection.
 * @return A map of strings to tensors containing the intersection results.
 */
std::map<std::string, torch::Tensor> Intersection(torch::Tensor* A, torch::Tensor* B, double nulls); 

/**
 * @brief Solves for neutrino momentum.
 *
 * @param H Pointer to the tensor containing event data.
 * @param sigma Pointer to the tensor containing uncertainties.
 * @param met_xy Pointer to the tensor containing MET x and y components.
 * @param null Null value for the computation.
 * @return A map of strings to tensors containing the neutrino momentum solutions.
 */
std::map<std::string, torch::Tensor> Nu(torch::Tensor* H, torch::Tensor* sigma, torch::Tensor* met_xy, double null); 

/**
 * @brief Solves for neutrino momenta in a two-neutrino system.
 *
 * @param H1_ Pointer to the tensor containing the first neutrino's event data.
 * @param H1_perp Pointer to the tensor containing the first neutrino's perpendicular data.
 * @param H2_ Pointer to the tensor containing the second neutrino's event data.
 * @param H2_perp Pointer to the tensor containing the second neutrino's perpendicular data.
 * @param met_xy Pointer to the tensor containing MET x and y components.
 * @param null Null value for the computation.
 * @param step Step size for the iterative solution.
 * @param tolerance Tolerance for the solution.
 * @param timeout Maximum number of iterations.
 * @return A map of strings to tensors containing the neutrino momenta solutions.
 */
std::map<std::string, torch::Tensor> NuNu(
    torch::Tensor* H1_, torch::Tensor* H1_perp, torch::Tensor* H2_, torch::Tensor* H2_perp, torch::Tensor* met_xy, 
    double null = 10e-10, const double step = 1e-9, const double tolerance = 1e-6, const unsigned int timeout = 1000
); 

/**
 * @brief Solves for neutrino momentum with additional mass and sigma information.
 *
 * @param pmc_b Pointer to the tensor containing b-quark momenta.
 * @param pmc_mu Pointer to the tensor containing muon momenta.
 * @param met_xy Pointer to the tensor containing MET x and y components.
 * @param masses Pointer to the tensor containing mass values.
 * @param sigma Pointer to the tensor containing uncertainties.
 * @param null Null value for the computation.
 * @return A map of strings to tensors containing the neutrino momentum solutions.
 */
std::map<std::string, torch::Tensor> Nu(
    torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, torch::Tensor* masses, torch::Tensor* sigma , double null
); 

/**
 * @brief Solves for neutrino momentum with specific mass values.
 *
 * @param pmc_b Pointer to the tensor containing b-quark momenta.
 * @param pmc_mu Pointer to the tensor containing muon momenta.
 * @param met_xy Pointer to the tensor containing MET x and y components.
 * @param sigma Pointer to the tensor containing uncertainties.
 * @param null Null value for the computation.
 * @param massT Top quark mass.
 * @param massW W boson mass.
 * @return A map of strings to tensors containing the neutrino momentum solutions.
 */
std::map<std::string, torch::Tensor> Nu(
    torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, torch::Tensor* sigma , double null, double massT, double massW
); 

/**
 * @brief Solves for neutrino momenta in a two-neutrino system with mass tensors.
 *
 * @param pmc_b1 Pointer to the tensor containing the first b-quark's momenta.
 * @param pmc_b2 Pointer to the tensor containing the second b-quark's momenta.
 * @param pmc_mu1 Pointer to the tensor containing the first muon's momenta.
 * @param pmc_mu2 Pointer to the tensor containing the second muon's momenta.
 * @param met_xy Pointer to the tensor containing MET x and y components.
 * @param null Null value for the computation.
 * @param m1 Pointer to the tensor containing the first mass values.
 * @param m2 Pointer to the tensor containing the second mass values (optional).
 * @param step Step size for the iterative solution.
 * @param tolerance Tolerance for the solution.
 * @param timeout Maximum number of iterations.
 * @return A map of strings to tensors containing the neutrino momenta solutions.
 */
std::map<std::string, torch::Tensor> NuNu(
    torch::Tensor* pmc_b1, torch::Tensor* pmc_b2, torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
    torch::Tensor* met_xy, double null, torch::Tensor* m1, torch::Tensor* m2 = nullptr, 
    const double step = 1e-9, const double tolerance = 1e-6, const unsigned int timeout = 1000
); 

/**
 * @brief Solves for neutrino momenta in a two-neutrino system with specific mass values.
 *
 * @param pmc_b1 Pointer to the tensor containing the first b-quark's momenta.
 * @param pmc_b2 Pointer to the tensor containing the second b-quark's momenta.
 * @param pmc_mu1 Pointer to the tensor containing the first muon's momenta.
 * @param pmc_mu2 Pointer to the tensor containing the second muon's momenta.
 * @param met_xy Pointer to the tensor containing MET x and y components.
 * @param null Null value for the computation.
 * @param massT1 Top quark mass for the first neutrino.
 * @param massW1 W boson mass for the first neutrino.
 * @param massT2 Top quark mass for the second neutrino.
 * @param massW2 W boson mass for the second neutrino.
 * @return A map of strings to tensors containing the neutrino momenta solutions.
 */
std::map<std::string, torch::Tensor> NuNu(
    torch::Tensor* pmc_b1, torch::Tensor* pmc_b2, torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
    torch::Tensor* met_xy, double null, double massT1, double massW1, double massT2, double massW2
); 

} // namespace nusol_

#endif // CU_NUSOL_BASE_H
