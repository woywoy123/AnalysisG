/**
 * @brief Compute P2
 * @param pmc Input tensor
 * @return Result tensor
 */
torch::Tensor physics_::P2(torch::Tensor* pmc);

/**
 * @brief Compute P
 * @param pmc Input tensor
 * @return Result tensor
 */
torch::Tensor physics_::P(torch::Tensor* pmc);

/**
 * @brief Compute Beta2
 * @param pmc Input tensor
 * @return Result tensor
 */
torch::Tensor physics_::Beta2(torch::Tensor* pmc);

/**
 * @brief Compute Beta
 * @param pmc Input tensor
 * @return Result tensor
 */
torch::Tensor physics_::Beta(torch::Tensor* pmc);

/**
 * @brief Compute M2
 * @param pmc Input tensor
 * @return Result tensor
 */
torch::Tensor physics_::M2(torch::Tensor* pmc);

/**
 * @brief Compute M
 * @param pmc Input tensor
 * @return Result tensor
 */
torch::Tensor physics_::M(torch::Tensor* pmc);

/**
 * @brief Compute Mt2
 * @param pmc Input tensor
 * @return Result tensor
 */
torch::Tensor physics_::Mt2(torch::Tensor* pmc);

/**
 * @brief Compute Mt
 * @param pmc Input tensor
 * @return Result tensor
 */
torch::Tensor physics_::Mt(torch::Tensor* pmc);

/**
 * @brief Compute Theta
 * @param pmc Input tensor
 * @return Result tensor
 */
torch::Tensor physics_::Theta(torch::Tensor* pmc);

/**
 * @brief Compute DeltaR from two 4-vectors
 * @param pmu1 First input tensor
 * @param pmu2 Second input tensor
 * @return Result tensor
 */
torch::Tensor physics_::DeltaR(torch::Tensor* pmu1, torch::Tensor* pmu2);

/**
 * @brief Compute DeltaR from eta and phi
 * @param eta1 First eta tensor
 * @param eta2 Second eta tensor
 * @param phi1 First phi tensor
 * @param phi2 Second phi tensor
 * @return Result tensor
 */
torch::Tensor physics_::DeltaR(torch::Tensor* eta1, torch::Tensor* eta2, torch::Tensor* phi1, torch::Tensor* phi2);

/**
 * @brief Compute P2 from px, py, pz
 * @param px px tensor
 * @param py py tensor
 * @param pz pz tensor
 * @return Result tensor
 */
torch::Tensor physics_::P2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

/**
 * @brief Compute P from px, py, pz
 * @param px px tensor
 * @param py py tensor
 * @param pz pz tensor
 * @return Result tensor
 */
torch::Tensor physics_::P(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

/**
 * @brief Compute Beta2 from px, py, pz, e
 * @param px px tensor
 * @param py py tensor
 * @param pz pz tensor
 * @param e e tensor
 * @return Result tensor
 */
torch::Tensor physics_::Beta2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Compute Beta from px, py, pz, e
 * @param px px tensor
 * @param py py tensor
 * @param pz pz tensor
 * @param e e tensor
 * @return Result tensor
 */
torch::Tensor physics_::Beta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Compute M2 from px, py, pz, e
 * @param px px tensor
 * @param py py tensor
 * @param pz pz tensor
 * @param e e tensor
 * @return Result tensor
 */
torch::Tensor physics_::M2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Compute M from px, py, pz, e
 * @param px px tensor
 * @param py py tensor
 * @param pz pz tensor
 * @param e e tensor
 * @return Result tensor
 */
torch::Tensor physics_::M(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Compute Mt2 from pz, e
 * @param pz pz tensor
 * @param e e tensor
 * @return Result tensor
 */
torch::Tensor physics_::Mt2(torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Compute Mt from pz, e
 * @param pz pz tensor
 * @param e e tensor
 * @return Result tensor
 */
torch::Tensor physics_::Mt(torch::Tensor* pz, torch::Tensor* e);

/**
 * @brief Compute Theta from px, py, pz
 * @param px px tensor
 * @param py py tensor
 * @param pz pz tensor
 * @return Result tensor
 */
torch::Tensor physics_::Theta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);
