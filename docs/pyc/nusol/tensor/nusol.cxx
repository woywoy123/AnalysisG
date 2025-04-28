/**
 * @file nusol.cxx
 * @brief This file contains the implementation of the nusol namespace, which provides functions for neutrino reconstruction.
 */

#include <operators/operators.h>
#include <transform/transform.h>
#include <physics/physics.h>
#include <utils/utils.h>
#include <nusol/nusol.h>

/**
 * @brief Generates a map of squared mass values from input tensors.
 * 
 * This function takes a tensor L and a tensor of masses, and computes the square of each mass component.
 * It returns a map containing the squared values for 'T2', 'W2', and 'N2'.
 *
 * @param L Pointer to a torch::Tensor.
 * @param masses Pointer to a torch::Tensor containing mass values.
 * @return std::map<std::string, torch::Tensor> A map containing the squared mass values.
 */
std::map<std::string, torch::Tensor> GetMasses(torch::Tensor* L, torch::Tensor* masses);

/**
 * @brief Calculates x0 value based on input tensors.
 * 
 * This function computes the x0 value using the provided pmc, _pm2, mH2, and mL2 tensors.
 *
 * @param pmc Pointer to a torch::Tensor.
 * @param _pm2 Pointer to a torch::Tensor.
 * @param mH2 Pointer to a torch::Tensor.
 * @param mL2 Pointer to a torch::Tensor.
 * @return torch::Tensor The computed x0 value.
 */
torch::Tensor _x0(torch::Tensor* pmc, torch::Tensor* _pm2, torch::Tensor* mH2, torch::Tensor* mL2);

/**
 * @brief Constructs a tensor representing horizontal and vertical components from input tensor G.
 * 
 * This function extracts specific elements from the input tensor G and arranges them into a new tensor
 * representing horizontal and vertical components.
 *
 * @param G Pointer to a torch::Tensor.
 * @return torch::Tensor A tensor representing horizontal and vertical components.
 */
torch::Tensor HorizontalVertical(torch::Tensor* G);

/**
 * @brief Computes a tensor representing parallel components based on input tensors G and CoF.
 * 
 * This function calculates a tensor representing parallel components using the provided G and CoF tensors.
 * It handles cases where g00 is 0 or greater than 0.
 *
 * @param G Pointer to a torch::Tensor.
 * @param CoF Pointer to a torch::Tensor.
 * @return torch::Tensor A tensor representing parallel components.
 */
torch::Tensor Parallel(torch::Tensor* G, torch::Tensor* CoF);

/**
 * @brief Computes a tensor representing intersecting components based on input tensors G, g22, and CoF.
 * 
 * This function calculates a tensor representing intersecting components using the provided G, g22, and CoF tensors.
 * It handles cases where -g22 is less than 0, equal to 0, or greater than 0.
 *
 * @param G Pointer to a torch::Tensor.
 * @param g22 Pointer to a torch::Tensor.
 * @param CoF Pointer to a torch::Tensor.
 * @return torch::Tensor A tensor representing intersecting components.
 */
torch::Tensor Intersecting(torch::Tensor* G, torch::Tensor* g22, torch::Tensor* CoF);

/**
 * @brief Computes Pi/2 tensor with the same dimensions as input tensor x.
 * 
 * This function creates a tensor filled with Pi/2 values, with the same dimensions as the input tensor x.
 *
 * @param x Pointer to a torch::Tensor.
 * @return torch::Tensor A tensor filled with Pi/2 values.
 */
torch::Tensor Pi_2(torch::Tensor* x);

/**
 * @brief Computes a rotation matrix based on input tensors pmc_b, pmc_mu, and base.
 * 
 * This function calculates a rotation matrix using the provided pmc_b, pmc_mu, and base tensors.
 *
 * @param pmc_b Pointer to a torch::Tensor.
 * @param pmc_mu Pointer to a torch::Tensor.
 * @param base Pointer to a torch::Tensor.
 * @return torch::Tensor The computed rotation matrix.
 */
torch::Tensor Rotation(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* base);

/**
 * @brief Creates a shape tensor based on input tensor x and a vector of diagonal values.
 * 
 * This function generates a shape tensor using the provided input tensor x and a vector of diagonal values.
 *
 * @param x Pointer to a torch::Tensor.
 * @param diag std::vector<int> A vector of diagonal values.
 * @return torch::Tensor The created shape tensor.
 */
torch::Tensor Shape(torch::Tensor* x, std::vector<int> diag);

/**
 * @brief Computes a sigma tensor based on input tensors x and sigma.
 * 
 * This function calculates a sigma tensor using the provided input tensors x and sigma.
 *
 * @param x Pointer to a torch::Tensor.
 * @param sigma Pointer to a torch::Tensor.
 * @return torch::Tensor The computed sigma tensor.
 */
torch::Tensor Sigma(torch::Tensor* x, torch::Tensor* sigma);

/**
 * @brief Transforms a missing transverse energy tensor.
 * 
 * This function transforms a missing transverse energy tensor met_xy into a 3x3 tensor.
 *
 * @param met_xy Pointer to a torch::Tensor representing missing transverse energy.
 * @return torch::Tensor The transformed tensor.
 */
torch::Tensor _met(torch::Tensor* met_xy);

/**
 * @brief Computes the perpendicular component of a base tensor.
 * 
 * This function calculates the perpendicular component of the input base tensor.
 *
 * @param base Pointer to a torch::Tensor.
 * @return torch::Tensor The perpendicular component of the base tensor.
 */
torch::Tensor _H_perp(torch::Tensor* base);

/**
 * @brief Computes a tensor N based on the perpendicular component of H.
 * 
 * This function calculates a tensor N using the provided perpendicular component of H.
 *
 * @param hperp Pointer to a torch::Tensor representing the perpendicular component of H.
 * @return torch::Tensor The computed tensor N.
 */
torch::Tensor _N(torch::Tensor* hperp);

namespace nusol_ {

/**
 * @brief Computes the base matrix for neutrino reconstruction.
 * 
 * This function calculates the base matrix used in neutrino reconstruction, using input tensors
 * representing particle momenta and masses.
 *
 * @param pmc_b Pointer to a torch::Tensor representing the momentum of the b-quark.
 * @param pmc_mu Pointer to a torch::Tensor representing the momentum of the muon.
 * @param masses Pointer to a torch::Tensor representing the masses of the particles.
 * @return torch::Tensor The computed base matrix.
 */
torch::Tensor BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses);

/**
 * @brief Computes the perpendicular component of H for neutrino reconstruction.
 * 
 * This function calculates the perpendicular component of H, used in neutrino reconstruction,
 * using input tensors representing particle momenta and masses.
 *
 * @param pmc_b Pointer to a torch::Tensor representing the momentum of the b-quark.
 * @param pmc_mu Pointer to a torch::Tensor representing the momentum of the muon.
 * @param masses Pointer to a torch::Tensor representing the masses of the particles.
 * @return torch::Tensor The computed perpendicular component of H.
 */
torch::Tensor Hperp(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses);

/**
 * @brief Computes the intersection solutions between two 3x3 matrices (tensors) A and B.
 *
 * This function finds the intersection points of two geometric objects represented by the input tensors A and B.
 * It handles various cases such as horizontal/vertical, parallel, and intersecting solutions, and applies
 * numerical stability improvements. The function also sorts and filters the solutions based on a numerical threshold.
 *
 * @param[in]  A    Pointer to the first input tensor (expected shape: [N, 3, 3]).
 * @param[in]  B    Pointer to the second input tensor (expected shape: [N, 3, 3]).
 * @param[in]  null Numerical threshold for filtering false solutions.
 * @return     std::tuple<torch::Tensor, torch::Tensor>
 *             - First tensor: The intersection solutions (shape: [N, M, 3]), where M is the number of valid solutions.
 *             - Second tensor: Diagnostic values for each solution (shape: [N, M]), indicating the quality of each solution.
 *
 * @note The function assumes that the input tensors are batched (i.e., the first dimension is the batch size).
 * @note The function internally clones the input tensors and does not modify the originals.
 * @note Solutions with diagnostic values greater than the threshold 'null' are filtered out.
 */
std::map<std::string, torch::Tensor> _xNu(
    torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, 
    torch::Tensor* masses, torch::Tensor* sigma
);

/**
 * @brief Computes neutrino solutions based on input tensors and parameters.
 * 
 * This function calculates the neutrino solutions using input tensors 
 * representing particle momenta, missing transverse energy, and other 
 * parameters. It returns a map containing the computed solutions and 
 * intermediate results.
 */
std::map<std::string, torch::Tensor> Nu(
    torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, 
    torch::Tensor* masses, torch::Tensor* sigma, double null
);

/**
 * @brief Computes neutrino solutions based on input tensors and parameters.
 * 
 * This function calculates the neutrino solutions using input tensors 
 * representing particle momenta, missing transverse energy, and other 
 * parameters. It returns a map containing the computed solutions and 
 * intermediate results.
 * 
 * @param pmc_b1 Pointer to a torch::Tensor representing the momentum of the first b-quark.
 * @param pmc_b2 Pointer to a torch::Tensor representing the momentum of the second b-quark.
 * @param pmc_l1 Pointer to a torch::Tensor representing the momentum of the first lepton.
 * @param pmc_l2 Pointer to a torch::Tensor representing the momentum of the second lepton.
 * @param met_xy Pointer to a torch::Tensor representing the missing transverse energy in x and y directions.
 * @param null A double value used as a parameter in the intersection calculation.
 * @param m1 Pointer to a torch::Tensor representing the mass of the first particle.
 * @param m2 Pointer to a torch::Tensor representing the mass of the second particle (optional, defaults to m1 if not provided).
 * 
 * @return A std::map<std::string, torch::Tensor> containing the following keys:
 *         - "NoSols": A tensor indicating whether solutions exist.
 *         - "n_": A tensor representing the computed n_ matrix.
 *         - "nu_1": A tensor representing the first neutrino solution.
 *         - "nu_2": A tensor representing the second neutrino solution.
 *         - "distance": A tensor representing the distance metric for solutions.
 *         - "H_perp_1": A tensor representing the perpendicular component of H1.
 *         - "H_perp_2": A tensor representing the perpendicular component of H2.
 */
std::map<std::string, torch::Tensor> NuNu(
    torch::Tensor* pmc_b1, torch::Tensor* pmc_b2, 
    torch::Tensor* pmc_l1, torch::Tensor* pmc_l2, 
    torch::Tensor* met_xy, double null, torch::Tensor* m1, torch::Tensor* m2);

} // namespace nusol_
