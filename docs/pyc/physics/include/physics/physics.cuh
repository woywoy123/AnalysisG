/**
 * @brief Computes squared momentum.
 * @param px Pointer to x-component of momentum.
 * @param py Pointer to y-component of momentum.
 * @param pz Pointer to z-component of momentum.
 * @return Tensor with computed P2.
 */
torch::Tensor P2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

/**
 * @brief Computes squared momentum from a 4-vector.
 * @param pmc Pointer to momentum 4-vector.
 * @return Tensor with computed P2.
 */
torch::Tensor P2(torch::Tensor* pmc);

/**
 * @brief Computes momentum.
 * @param px Pointer to x-component of momentum.
 * @param py Pointer to y-component of momentum.
 * @param pz Pointer to z-component of momentum.
 * @return Tensor with computed P.
 */
torch::Tensor P(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

/**
 * @brief Computes momentum from a 4-vector.
 * @param pmc Pointer to momentum 4-vector.
 * @return Tensor with computed P.
 */
torch::Tensor P(torch::Tensor* pmc);

/**
 * @brief Computes beta squared.
 * @param px Pointer to x-component of momentum.
 * @param py Pointer to y-component of momentum.
 * @param pz Pointer to z-component of momentum.
 * @param e Pointer to energy component.
 * @return Tensor with computed beta squared.
 */
torch::Tensor Beta2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes beta squared from a 4-vector.
 * @param pmc Pointer to momentum 4-vector.
 * @return Tensor with computed beta squared.
 */
torch::Tensor Beta2(torch::Tensor* pmc);

/**
 * @brief Computes beta.
 * @param px Pointer to x-component of momentum.
 * @param py Pointer to y-component of momentum.
 * @param pz Pointer to z-component of momentum.
 * @param e Pointer to energy component.
 * @return Tensor with computed beta.
 */
torch::Tensor Beta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes beta from a 4-vector.
 * @param pmc Pointer to momentum 4-vector.
 * @return Tensor with computed beta.
 */
torch::Tensor Beta(torch::Tensor* pmc);

/**
 * @brief Computes invariant mass squared.
 * @param px Pointer to x-component of momentum.
 * @param py Pointer to y-component of momentum.
 * @param pz Pointer to z-component of momentum.
 * @param e Pointer to energy component.
 * @return Tensor with computed mass squared.
 */
torch::Tensor M2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes invariant mass squared from a 4-vector.
 * @param pmc Pointer to momentum 4-vector.
 * @return Tensor with computed mass squared.
 */
torch::Tensor M2(torch::Tensor* pmc);

/**
 * @brief Computes invariant mass.
 * @param px Pointer to x-component of momentum.
 * @param py Pointer to y-component of momentum.
 * @param pz Pointer to z-component of momentum.
 * @param e Pointer to energy component.
 * @return Tensor with computed mass.
 */
torch::Tensor M(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes invariant mass from a 4-vector.
 * @param pmc Pointer to momentum 4-vector.
 * @return Tensor with computed mass.
 */
torch::Tensor M(torch::Tensor* pmc);

/**
 * @brief Computes transverse mass squared.
 * @param pz Pointer to z-component of momentum.
 * @param e Pointer to energy component.
 * @return Tensor with computed transverse mass squared.
 */
torch::Tensor Mt2(torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes transverse mass squared from a 4-vector.
 * @param pmc Pointer to momentum 4-vector.
 * @return Tensor with computed transverse mass squared.
 */
torch::Tensor Mt2(torch::Tensor* pmc);

/**
 * @brief Computes transverse mass.
 * @param pz Pointer to z-component of momentum.
 * @param e Pointer to energy component.
 * @return Tensor with computed transverse mass.
 */
torch::Tensor Mt(torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Computes transverse mass from a 4-vector.
 * @param pmc Pointer to momentum 4-vector.
 * @return Tensor with computed transverse mass.
 */
torch::Tensor Mt(torch::Tensor* pmc);

/**
 * @brief Computes the polar angle theta from a 4-vector.
 * @param pmc Pointer to momentum 4-vector.
 * @return Tensor with computed theta.
 */
torch::Tensor Theta(torch::Tensor* pmc);

/**
 * @brief Computes the polar angle theta from components.
 * @param px Pointer to x-component of momentum.
 * @param py Pointer to y-component of momentum.
 * @param pz Pointer to z-component of momentum.
 * @return Tensor with computed theta.
 */
torch::Tensor Theta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

/**
 * @brief Computes DeltaR for two 4-vectors.
 * @param pmu1 Pointer to first 4-vector.
 * @param pmu2 Pointer to second 4-vector.
 * @return Tensor with computed DeltaR.
 */
torch::Tensor DeltaR(torch::Tensor* pmu1, torch::Tensor* pmu2);

/**
 * @brief Computes DeltaR from eta and phi values.
 * @param eta1 Pointer to pseudorapidity of first particle.
 * @param eta2 Pointer to pseudorapidity of second particle.
 * @param phi1 Pointer to azimuthal angle of first particle.
 * @param phi2 Pointer to azimuthal angle of second particle.
 * @return Tensor with computed DeltaR.
 */
torch::Tensor DeltaR(torch::Tensor* eta1, torch::Tensor* eta2, torch::Tensor* phi1, torch::Tensor* phi2);
