/**
 * @brief Computes P2 from separate momentum components.
 * @param px Pointer to x-component of momentum.
 * @param py Pointer to y-component of momentum.
 * @param pz Pointer to z-component of momentum.
 * @return Tensor containing the computed P2.
 */
torch::Tensor P2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

/**
 * @brief Computes P2 from a combined momentum tensor.
 * @param pmc Pointer to momentum tensor (px, py, pz, E).
 * @return Tensor containing the computed P2.
 */
torch::Tensor P2(torch::Tensor* pmc);

/**
 * @brief Computes P from separate momentum components.
 * @param px Pointer to x-component of momentum.
 * @param py Pointer to y-component of momentum.
 * @param pz Pointer to z-component of momentum.
 * @return Tensor containing the computed P.
 */
torch::Tensor P(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

/**
 * @brief Computes P from a combined momentum tensor.
 * @param pmc Pointer to momentum tensor (px, py, pz, E).
 * @return Tensor containing the computed P.
 */
torch::Tensor P(torch::Tensor* pmc);

/**
 * @brief Computes Beta^2 from separate momentum components and energy.
 * @param px Pointer to x-component of momentum.
 * @param py Pointer to y-component of momentum.
 * @param pz Pointer to z-component of momentum.
 * @param e Pointer to energy component.
 * @return Tensor containing the computed Beta^2.
 */
torch::Tensor Beta2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes Beta^2 from a combined momentum tensor.
 * @param pmc Pointer to momentum tensor (px, py, pz, E).
 * @return Tensor containing the computed Beta^2.
 */
torch::Tensor Beta2(torch::Tensor* pmc);

/**
 * @brief Computes Beta from separate momentum components and energy.
 * @param px Pointer to x-component of momentum.
 * @param py Pointer to y-component of momentum.
 * @param pz Pointer to z-component of momentum.
 * @param e Pointer to energy component.
 * @return Tensor containing the computed Beta.
 */
torch::Tensor Beta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes Beta from a combined momentum tensor.
 * @param pmc Pointer to momentum tensor (px, py, pz, E).
 * @return Tensor containing the computed Beta.
 */
torch::Tensor Beta(torch::Tensor* pmc);

/**
 * @brief Computes M^2 from separate momentum components and energy.
 * @param px Pointer to x-component of momentum.
 * @param py Pointer to y-component of momentum.
 * @param pz Pointer to z-component of momentum.
 * @param e Pointer to energy component.
 * @return Tensor containing the computed M^2.
 */
torch::Tensor M2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes M^2 from a combined momentum tensor.
 * @param pmc Pointer to momentum tensor (px, py, pz, E).
 * @return Tensor containing the computed M^2.
 */
torch::Tensor M2(torch::Tensor* pmc);

/**
 * @brief Computes M from separate momentum components and energy.
 * @param px Pointer to x-component of momentum.
 * @param py Pointer to y-component of momentum.
 * @param pz Pointer to z-component of momentum.
 * @param e Pointer to energy component.
 * @return Tensor containing the computed M.
 */
torch::Tensor M(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes M from a combined momentum tensor.
 * @param pmc Pointer to momentum tensor (px, py, pz, E).
 * @return Tensor containing the computed M.
 */
torch::Tensor M(torch::Tensor* pmc);

/**
 * @brief Computes M_t^2 from z-momentum and energy.
 * @param pz Pointer to z-component of momentum.
 * @param e Pointer to energy component.
 * @return Tensor containing the computed M_t^2.
 */
torch::Tensor Mt2(torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes M_t^2 from a combined momentum tensor.
 * @param pmc Pointer to momentum tensor (px, py, pz, E).
 * @return Tensor containing the computed M_t^2.
 */
torch::Tensor Mt2(torch::Tensor* pmc);

/**
 * @brief Computes M_t from z-momentum and energy.
 * @param pz Pointer to z-component of momentum.
 * @param e Pointer to energy component.
 * @return Tensor containing the computed M_t.
 */
torch::Tensor Mt(torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes M_t from a combined momentum tensor.
 * @param pmc Pointer to momentum tensor (px, py, pz, E).
 * @return Tensor containing the computed M_t.
 */
torch::Tensor Mt(torch::Tensor* pmc);

/**
 * @brief Computes Theta from a combined momentum tensor.
 * @param pmc Pointer to momentum tensor (px, py, pz, E).
 * @return Tensor containing the computed angle.
 */
torch::Tensor Theta(torch::Tensor* pmc);

/**
 * @brief Computes Theta from separate momentum components.
 * @param px Pointer to x-component of momentum.
 * @param py Pointer to y-component of momentum.
 * @param pz Pointer to z-component of momentum.
 * @return Tensor containing the computed angle.
 */
torch::Tensor Theta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

/**
 * @brief Computes DeltaR between two four-momentum tensors.
 * @param pmu1 Pointer to first four-momentum tensor (eta, phi, etc.).
 * @param pmu2 Pointer to second four-momentum tensor.
 * @return Tensor containing the computed DeltaR.
 */
torch::Tensor DeltaR(torch::Tensor* pmu1, torch::Tensor* pmu2);

/**
 * @brief Computes DeltaR from separate eta and phi values.
 * @param eta1 Pointer to first eta value.
 * @param eta2 Pointer to second eta value.
 * @param phi1 Pointer to first phi value.
 * @param phi2 Pointer to second phi value.
 * @return Tensor containing the computed DeltaR.
 */
torch::Tensor DeltaR(torch::Tensor* eta1, torch::Tensor* eta2, torch::Tensor* phi1, torch::Tensor* phi2);
