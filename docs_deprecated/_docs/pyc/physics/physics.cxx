/**
 * @brief Compute the squared momentum from a 4-vector.
 * @param pmc Pointer to the input 4-vector.
 * @return Tensor containing the squared momentum.
 */
torch::Tensor physics_::P2(torch::Tensor* pmc);

/**
 * @brief Compute the squared momentum from separate components.
 * @param px Pointer to the x-component.
 * @param py Pointer to the y-component.
 * @param pz Pointer to the z-component.
 * @return Tensor containing the squared momentum.
 */
torch::Tensor physics_::P2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

/**
 * @brief Compute the momentum from a 4-vector.
 * @param pmc Pointer to the input 4-vector.
 * @return Tensor containing the momentum.
 */
torch::Tensor physics_::P(torch::Tensor* pmc);

/**
 * @brief Compute the momentum from separate components.
 * @param px Pointer to the x-component.
 * @param py Pointer to the y-component.
 * @param pz Pointer to the z-component.
 * @return Tensor containing the momentum.
 */
torch::Tensor physics_::P(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

/**
 * @brief Compute the squared beta from a 4-vector.
 * @param pmc Pointer to the input 4-vector.
 * @return Tensor containing beta squared.
 */
torch::Tensor physics_::Beta2(torch::Tensor* pmc);

/**
 * @brief Compute the squared beta from separate components.
 * @param px Pointer to the x-component.
 * @param py Pointer to the y-component.
 * @param pz Pointer to the z-component.
 * @param e Pointer to the energy component.
 * @return Tensor containing beta squared.
 */
torch::Tensor physics_::Beta2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Compute beta from a 4-vector.
 * @param pmc Pointer to the input 4-vector.
 * @return Tensor containing beta.
 */
torch::Tensor physics_::Beta(torch::Tensor* pmc);

/**
 * @brief Compute beta from separate components.
 * @param px Pointer to the x-component.
 * @param py Pointer to the y-component.
 * @param pz Pointer to the z-component.
 * @param e Pointer to the energy component.
 * @return Tensor containing beta.
 */
torch::Tensor physics_::Beta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Compute the invariant mass squared from a 4-vector.
 * @param pmc Pointer to the input 4-vector.
 * @return Tensor containing the mass squared.
 */
torch::Tensor physics_::M2(torch::Tensor* pmc);

/**
 * @brief Compute the invariant mass squared from separate components.
 * @param px Pointer to the x-component.
 * @param py Pointer to the y-component.
 * @param pz Pointer to the z-component.
 * @param e Pointer to the energy component.
 * @return Tensor containing the mass squared.
 */
torch::Tensor physics_::M2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Compute the invariant mass from a 4-vector.
 * @param pmc Pointer to the input 4-vector.
 * @return Tensor containing the mass.
 */
torch::Tensor physics_::M(torch::Tensor* pmc);

/**
 * @brief Compute the invariant mass from separate components.
 * @param px Pointer to the x-component.
 * @param py Pointer to the y-component.
 * @param pz Pointer to the z-component.
 * @param e Pointer to the energy component.
 * @return Tensor containing the mass.
 */
torch::Tensor physics_::M(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Compute the transverse mass squared from a 4-vector.
 * @param pmc Pointer to the input 4-vector.
 * @return Tensor containing the transverse mass squared.
 */
torch::Tensor physics_::Mt2(torch::Tensor* pmc);

/**
 * @brief Compute the transverse mass squared from components.
 * @param pz Pointer to the z-component.
 * @param e Pointer to the energy component.
 * @return Tensor containing the transverse mass squared.
 */
torch::Tensor physics_::Mt2(torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Compute the transverse mass from a 4-vector.
 * @param pmc Pointer to the input 4-vector.
 * @return Tensor containing the transverse mass.
 */
torch::Tensor physics_::Mt(torch::Tensor* pmc);

/**
 * @brief Compute the transverse mass from components.
 * @param pz Pointer to the z-component.
 * @param e Pointer to the energy component.
 * @return Tensor containing the transverse mass.
 */
torch::Tensor physics_::Mt(torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Compute the polar angle from a 4-vector.
 * @param pmc Pointer to the input 4-vector.
 * @return Tensor containing the polar angle.
 */
torch::Tensor physics_::Theta(torch::Tensor* pmc);

/**
 * @brief Compute the polar angle from separate components.
 * @param px Pointer to the x-component.
 * @param py Pointer to the y-component.
 * @param pz Pointer to the z-component.
 * @return Tensor containing the polar angle.
 */
torch::Tensor physics_::Theta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

/**
 * @brief Compute the distance ΔR between two 4-vectors.
 * @param pmu1 Pointer to the first 4-vector.
 * @param pmu2 Pointer to the second 4-vector.
 * @return Tensor containing ΔR.
 */
torch::Tensor physics_::DeltaR(torch::Tensor* pmu1, torch::Tensor* pmu2);

/**
 * @brief Compute the distance ΔR from separate parts.
 * @param eta1 Pointer to the first pseudorapidity.
 * @param eta2 Pointer to the second pseudorapidity.
 * @param phi1 Pointer to the first azimuthal angle.
 * @param phi2 Pointer to the second azimuthal angle.
 * @return Tensor containing ΔR.
 */
torch::Tensor physics_::DeltaR(torch::Tensor* eta1, torch::Tensor* eta2, torch::Tensor* phi1, torch::Tensor* phi2);
