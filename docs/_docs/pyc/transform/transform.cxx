/**
 * @file
 * @brief Provides functions to transform momentum components between different coordinate representations.
 */

/**
 * @brief Computes the x-component of momentum.
 * @param pt Pointer to a tensor containing transverse momentum.
 * @param phi Pointer to a tensor containing azimuthal angle.
 * @return Tensor with px values.
 */
torch::Tensor transform_::Px(torch::Tensor* pt, torch::Tensor* phi);

/**
 * @brief Computes the y-component of momentum.
 * @param pt Pointer to a tensor containing transverse momentum.
 * @param phi Pointer to a tensor containing azimuthal angle.
 * @return Tensor with py values.
 */
torch::Tensor transform_::Py(torch::Tensor* pt, torch::Tensor* phi);

/**
 * @brief Computes the z-component of momentum.
 * @param pt Pointer to a tensor containing transverse momentum.
 * @param eta Pointer to a tensor containing pseudorapidity.
 * @return Tensor with pz values.
 */
torch::Tensor transform_::Pz(torch::Tensor* pt, torch::Tensor* eta);

/**
 * @brief Computes (px, py, pz) from (pt, eta, phi).
 * @param pt Pointer to a tensor for transverse momentum.
 * @param eta Pointer to a tensor for pseudorapidity.
 * @param phi Pointer to a tensor for azimuthal angle.
 * @return Concatenated tensor [px, py, pz].
 */
torch::Tensor transform_::PxPyPz(torch::Tensor* pt, torch::Tensor* eta,
                                 torch::Tensor* phi);

/**
 * @brief Computes (px, py, pz, E) from (pt, eta, phi, E).
 * @param pt Pointer to the transverse momentum tensor.
 * @param eta Pointer to the pseudorapidity tensor.
 * @param phi Pointer to the azimuthal angle tensor.
 * @param e Pointer to the energy tensor.
 * @return Concatenated tensor [px, py, pz, E].
 */
torch::Tensor transform_::PxPyPzE(torch::Tensor* pt, torch::Tensor* eta,
                                  torch::Tensor* phi, torch::Tensor* e);

/**
 * @brief Transforms (pt, eta, phi, [E]) into (px, py, pz, [E]).
 * @param pmu Pointer to the momentum tensor, of shape (..., 3) or (..., 4).
 * @return Transformed tensor [px, py, pz] or [px, py, pz, E].
 */
torch::Tensor transform_::PxPyPz(torch::Tensor* pmu);

/**
 * @brief Computes transverse momentum from (px, py).
 * @param px Pointer to a tensor with px values.
 * @param py Pointer to a tensor with py values.
 * @return Tensor with pt.
 */
torch::Tensor transform_::Pt(torch::Tensor* px, torch::Tensor* py);

/**
 * @brief Computes azimuthal angle from (px, py).
 * @param px Pointer to a tensor with px values.
 * @param py Pointer to a tensor with py values.
 * @return Tensor with phi.
 */
torch::Tensor transform_::Phi(torch::Tensor* px, torch::Tensor* py);

/**
 * @brief Computes azimuthal angle from a tensor of shape (..., 2).
 * @param pmc Pointer to the momentum tensor.
 * @return Tensor with phi.
 */
torch::Tensor transform_::Phi(torch::Tensor* pmc);

/**
 * @brief Computes pseudorapidity from (pt, pz).
 * @param pt Pointer to a tensor with transverse momentum.
 * @param pz Pointer to a tensor with pz values.
 * @return Tensor with eta.
 */
torch::Tensor transform_::PtEta(torch::Tensor* pt, torch::Tensor* pz);

/**
 * @brief Computes pseudorapidity from (px, py, pz).
 * @param px Pointer to the px tensor.
 * @param py Pointer to the py tensor.
 * @param pz Pointer to the pz tensor.
 * @return Tensor with eta.
 */
torch::Tensor transform_::Eta(torch::Tensor* px, torch::Tensor* py,
                              torch::Tensor* pz);

/**
 * @brief Computes pseudorapidity from a tensor of shape (..., 3).
 * @param pmc Pointer to the momentum tensor.
 * @return Tensor with eta.
 */
torch::Tensor transform_::Eta(torch::Tensor* pmc);

/**
 * @brief Computes (pt, eta, phi) from (px, py, pz).
 * @param px Pointer to the px tensor.
 * @param py Pointer to the py tensor.
 * @param pz Pointer to the pz tensor.
 * @return Concatenated tensor [pt, eta, phi].
 */
torch::Tensor transform_::PtEtaPhi(torch::Tensor* px, torch::Tensor* py,
                                   torch::Tensor* pz);

/**
 * @brief Computes (pt, eta, phi, [E]) from (px, py, pz, [E]).
 * @param pmc Pointer to the momentum tensor, shape (..., 3) or (..., 4).
 * @return Tensor [pt, eta, phi] or [pt, eta, phi, E].
 */
torch::Tensor transform_::PtEtaPhi(torch::Tensor* pmc);

/**
 * @brief Computes (pt, eta, phi, E) from (px, py, pz, E).
 * @param px Pointer to the px tensor.
 * @param py Pointer to the py tensor.
 * @param pz Pointer to the pz tensor.
 * @param e Pointer to the energy tensor.
 * @return Concatenated tensor [pt, eta, phi, E].
 */
torch::Tensor transform_::PtEtaPhiE(torch::Tensor* px, torch::Tensor* py,
                                    torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes (pt, eta, phi, E) from a tensor of shape (..., 4).
 * @param pmc Pointer to the momentum tensor.
 * @return Tensor [pt, eta, phi, E].
 */
torch::Tensor transform_::PtEtaPhiE(torch::Tensor* pmc);
