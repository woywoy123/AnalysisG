/**
 * @file utils.cuh
 * @brief This file contains various CUDA utility kernel functions designed for high-performance 
 *        computations, particularly relevant in physics analysis contexts involving particle data, 
 *        graph structures, and numerical methods. It leverages CUDA parallelism and PyTorch tensor 
 *        accessors for efficient GPU processing.
 */

#ifndef CU_NUSOL_UTILS_H
#define CU_NUSOL_UTILS_H

#include <utils/atomic.cuh> // Includes atomic operations, likely used for safe parallel updates.
#include <torch/extension.h> // Includes PyTorch C++ extension headers for tensor accessors.

/**
 * @brief Counts the total number of events and the occurrences of specific particle IDs (PIDs) 
 *        within each distinct batch present in the input data.
 *
 * @details This CUDA kernel operates in parallel across the input events. It reads the batch index 
 *          for each event and atomically increments the total event count for that batch in the 
 *          `num_events` output tensor. It also inspects the PID information for each event 
 *          (presumably flags indicating particle types like leptons or b-quarks) and atomically 
 *          increments the corresponding counters in the `num_pid` output tensor for the respective batch.
 *          The use of atomic operations ensures thread-safe updates to the shared counters for each batch.
 *          The kernel is templated on the data type `scalar_t` to allow flexibility with different 
 *          numerical precisions (e.g., float, double).
 *
 * @tparam scalar_t The data type (e.g., float, double, int) of the elements in the input tensors. 
 *                  Must be compatible with the data stored in `batch` and `pid`.
 * @param[in] batch A 1D tensor where each element represents the batch index associated with an event. 
 *                  Shape: (num_total_events).
 * @param[in] pid A 2D tensor containing PID information for each event. The second dimension likely 
 *                holds flags or identifiers for specific particle types (e.g., column 0 for leptons, 
 *                column 1 for b-quarks). Shape: (num_total_events, num_pid_types).
 * @param[out] num_events A 1D tensor to store the computed total number of events for each batch. 
 *                        It should be pre-allocated and initialized (likely to zero) with a size 
 *                        equal to the maximum batch index + 1. Shape: (num_batches).
 * @param[out] num_pid A 2D tensor to store the computed counts of specific PIDs for each batch. 
 *                     It should be pre-allocated and initialized (likely to zero). The second dimension 
 *                     should match the number of PID types being counted. Shape: (num_batches, num_pid_types).
 * @param[in] dx The total number of events (elements in the `batch` and `pid` tensors) being processed. 
 *               This is used to define the range of the parallel iteration within the kernel launch.
 */
template <typename scalar_t>
__global__ void _count(
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> batch, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pid, 
          torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> num_events,
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> num_pid,
    const unsigned int dx
);

/**
 * @brief Identifies valid combinations of two leptons (l) and two b-quarks (b), denoted as llbb, 
 *        within events, considering particle types and graph connectivity.
 *
 * @details This kernel processes events, likely representing particle interactions structured as graphs. 
 *          It first filters events based on the pre-calculated counts of leptons and b-quarks per batch 
 *          (provided by `num_pid`), potentially requiring exactly two leptons and two b-quarks for an 
 *          event to be considered. For qualifying events, it examines the graph structure defined by 
 *          `edge_idx` and `num_edges` to find specific combinations of four particles (identified by 
 *          their indices) that match the llbb criteria based on their `pid` information. The indices 
 *          of the particles forming a valid llbb combination are stored in the `llbb` output tensor, 
 *          and a corresponding mask `msk` is set to indicate the validity of the combination found for 
 *          that event or potential combination slot. The kernel likely iterates through edges or nodes 
 *          within each event's graph structure.
 *
 * @tparam scalar_t The data type of the elements in the input tensors.
 * @param[in] pid A 2D tensor containing PID information for each particle/node in the graph. 
 *                Shape: (num_total_particles, num_pid_types).
 * @param[in] batch A 1D tensor mapping each particle/node to its corresponding batch index. 
 *                  Shape: (num_total_particles).
 * @param[in] num_edges A 1D tensor indicating the number of edges associated with each event or graph. 
 *                      Shape: (num_events).
 * @param[in] edge_idx A 2D tensor representing the graph edges. Each row typically contains the indices 
 *                     of the two connected particles/nodes. Shape: (n_edges, 2).
 * @param[in] num_pid A 2D tensor containing the pre-calculated counts of specific PIDs (leptons, b-quarks) 
 *                    per batch, used for initial filtering. Shape: (num_batches, num_pid_types).
 * @param[out] llbb A 2D tensor where each row will store the indices of the four particles forming a 
 *                  valid llbb combination. Shape: (num_potential_combinations, 4).
 * @param[out] msk A 1D tensor acting as a boolean mask, where a non-zero value indicates that a valid 
 *                 llbb combination was found and stored in the corresponding row of `llbb`. 
 *                 Shape: (num_potential_combinations).
 * @param[in] n_edges The total number of edges across all graphs/events, defining the size of `edge_idx`.
 * @param[in] mx The maximum number of edges expected for any single event/graph. This might be used 
 *               for indexing or memory allocation strategies within the kernel.
 */
template <typename scalar_t>
__global__ void _combination(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pid,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> batch,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> num_edges,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> edge_idx,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> num_pid,
      torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> llbb,
      torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> msk,
    const unsigned int n_edges, const unsigned mx
);

/**
 * @brief Calculates and populates a 2D mass matrix based on specified mass ranges and step counts.
 *
 * @details This kernel computes values for a mass matrix, likely used in subsequent physics calculations 
 *          or simulations. It appears to generate mass values based on ranges defined by `mTl`, `mTs` 
 *          (possibly for top quark mass) and `mWl`, `mWs` (possibly for W boson mass). The calculation 
 *          is performed over a grid or sequence defined by `steps`. Each thread likely calculates one 
 *          element or a set of elements of the `mass_` matrix. The exact formula for the calculation is 
 *          implemented within the kernel but not detailed here. It assumes `double` precision for the 
 *          calculations and the output matrix.
 *
 * @param[out] mass_ A 2D tensor where the calculated mass values will be stored. It should be 
 *                   pre-allocated with the appropriate dimensions, likely related to `steps`. 
 *                   Shape: (dimension1, dimension2), possibly (steps, steps) or similar.
 * @param[in] mTl Lower bound or starting value for the first mass parameter (e.g., top mass).
 * @param[in] mTs Step size or upper bound for the first mass parameter.
 * @param[in] mWl Lower bound or starting value for the second mass parameter (e.g., W mass).
 * @param[in] mWs Step size or upper bound for the second mass parameter.
 * @param[in] steps The number of steps or discretization points used in generating the mass values, 
 *                  likely defining the dimensions of the calculation grid.
 */
__global__ void _mass_matrix(
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> mass_,
    double mTl, double mTs, double mWl, double mWs, unsigned int steps
);

/**
 * @brief Performs a perturbation calculation, potentially related to numerical differentiation or 
 *        sensitivity analysis, using shared memory for intermediate data storage.
 *
 * @details This kernel executes a calculation involving several input tensors (`dnu_tw1`, `dnu_tw2`, 
 *          `dnu_met`, `dnu_res`), likely representing components of a system or derivatives. It applies 
 *          a perturbation factor (`perturb`) and uses physical constants (`top_mass`, `w_mass`). 
 *          The `start` flag might control initialization or specific calculation paths. A key feature 
 *          is the use of shared memory, indicated by the template parameter `size_x`. Data from the 
 *          input tensors is likely loaded into shared memory arrays within each thread block, calculations 
 *          are performed using this fast on-chip memory, and results might be written back to global 
 *          memory (though no output tensors are explicitly listed as parameters, the input tensors might 
 *          be updated, or this kernel is part of a sequence where results are used later). Shared memory 
 *          usage aims to reduce global memory access latency and improve performance.
 *
 * @tparam size_x A compile-time constant specifying the size of the shared memory arrays allocated per 
 *                thread block. This typically corresponds to the block dimension or a related size.
 * @param[in] dnu_tw1 A 2D tensor containing input data, possibly related to top/W decay products. 
 *                    Shape: (num_elements, dimension).
 * @param[in] dnu_tw2 A 2D tensor containing input data, similar to `dnu_tw1`. 
 *                    Shape: (num_elements, dimension).
 * @param[in] dnu_met A 2D tensor containing input data, possibly related to missing transverse energy (MET). 
 *                    Shape: (num_elements, dimension).
 * @param[in] dnu_res A 2D tensor containing input data, possibly representing residuals or results. 
 *                    Shape: (num_elements, dimension).
 * @param[in] perturb The magnitude or step size of the perturbation being applied.
 * @param[in] top_mass The value of the top quark mass used in the calculation.
 * @param[in] w_mass The value of the W boson mass used in the calculation.
 * @param[in] start A boolean flag, possibly indicating the start of an iterative process or a specific 
 *                  mode of operation.
 */
template <size_t size_x>
__global__ void _perturbation(
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> dnu_tw1,
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> dnu_tw2,
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> dnu_met,
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> dnu_res,
    const double perturb, const double top_mass, 
    const double w_mass, const bool start
);

/**
 * @brief Assigns specified top quark mass (`mass_t`) and W boson mass (`mass_w`) values to rows 
 *        of an output tensor based on the thread's y-index.
 *
 * @details This kernel initializes or sets values in the `mass_tw` tensor. It operates on a 2D grid 
 *          of threads. Threads with `threadIdx.y == 0` assign `mass_t` to their corresponding element(s) 
 *          in `mass_tw`, while threads with `threadIdx.y > 0` assign `mass_w`. This suggests `mass_tw` 
 *          might store mass hypotheses or parameters, potentially with the first row dedicated to the 
 *          top mass and subsequent rows to the W mass, replicated across the `lenx` dimension.
 *
 * @param[out] mass_tw A 2D tensor to be populated with mass values. Shape: (num_rows, lenx).
 * @param[in] mass_t The mass value (e.g., top quark mass) to assign when `threadIdx.y == 0`.
 * @param[in] mass_w The mass value (e.g., W boson mass) to assign when `threadIdx.y > 0`.
 * @param[in] lenx The size of the second dimension of the `mass_tw` tensor, determining how many 
 *                 columns are processed.
 */
__global__ void _assign_mass(
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> mass_tw,
    const double mass_t, const double mass_w, const long lenx
);

/**
 * @brief Applies a perturbation to input parameters and calculates the resulting changes in related quantities.
 *
 * @details This kernel takes input parameters (`nu_params`, possibly related to neutrino kinematics) 
 *          and applies a small change (`dt`) based on an offset (`ofs`). It then calculates the effect 
 *          of this perturbation on other quantities, storing the results in `dnu_met`, `dnu_tw1`, and 
 *          `dnu_tw2`. This is characteristic of numerical differentiation (finite differences) used to 
 *          compute gradients or Jacobians. Shared memory, sized by `size_x`, is likely used to cache 
 *          `nu_params` or intermediate results for efficient computation within a thread block. The `ofs` 
 *          parameter might indicate which component of `nu_params` is being perturbed in this specific call.
 *
 * @tparam size_x Compile-time constant for shared memory allocation size per block.
 * @param[in] nu_params A 2D tensor containing the input parameters to be perturbed. 
 *                      Shape: (lnx, num_params).
 * @param[out] dnu_met A 2D tensor to store the calculated change in the MET-related quantity due to the 
 *                     perturbation. Shape: (lnx, dimension_met).
 * @param[out] dnu_tw1 A 2D tensor to store the calculated change in the first top/W-related quantity. 
 *                     Shape: (lnx, dimension_tw1).
 * @param[out] dnu_tw2 A 2D tensor to store the calculated change in the second top/W-related quantity. 
 *                     Shape: (lnx, dimension_tw2).
 * @param[in] lnx The number of independent systems or data points being processed (size of the first 
 *                dimension).
 * @param[in] dt The step size or magnitude of the perturbation applied.
 * @param[in] ofs An offset, likely indicating which parameter dimension within `nu_params` is being 
 *                perturbed (e.g., 0 for the first parameter, 1 for the second, etc.).
 */
template <size_t size_x>
__global__ void _perturb(
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> nu_params,
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> dnu_met,
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> dnu_tw1,
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> dnu_tw2,
    const unsigned long lnx, const double dt, const unsigned int ofs
);

/**
 * @brief Performs one iteration of the Jacobi method to update parameters, likely for solving a system 
 *        of equations numerically.
 *
 * @details This kernel implements a step of the Jacobi iterative solver. It updates the `nu_params` 
 *          tensor based on its current values and the residual values provided in `dnu_res`. The update 
 *          likely involves calculating derivatives or using a pre-computed Jacobian (potentially derived 
 *          from kernels like `_perturb`) and applying the Jacobi update rule: 
 *          `x_new = D^-1 * (b - (L+U) * x_old)`. The `dt` parameter might be related to a learning rate 
 *          or relaxation factor, and `ofs` could potentially select a subset of parameters to update, 
 *          although in standard Jacobi, all are updated simultaneously using values from the previous 
 *          iteration. Shared memory (sized by `size_x`) is used to optimize data access during the 
 *          derivative calculation and update step within each thread block.
 *
 * @tparam size_x Compile-time constant for shared memory allocation size per block.
 * @param[in,out] nu_params A 2D tensor containing the parameters being iteratively updated. Values are 
 *                          read from the previous iteration and updated in place for the next iteration. 
 *                          Shape: (lnx, num_params).
 * @param[in] dnu_res A 2D tensor containing the residuals or error terms (e.g., `b - A*x_old`) used in 
 *                    the Jacobi update calculation. Shape: (lnx, num_residuals).
 * @param[in] lnx The number of independent systems or data points being processed.
 * @param[in] dt A parameter potentially related to a step size, learning rate, or relaxation factor 
 *               in the iterative update.
 * @param[in] ofs An offset parameter, its specific role might depend on the exact implementation 
 *                (e.g., selecting parameter subsets, indexing within shared memory).
 */
template <size_t size_x>
__global__ void _jacobi(
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> nu_params,
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> dnu_res,
    const unsigned long lnx, const double dt, unsigned int ofs
);

/**
 * @brief Compares multiple candidate solutions for each event and selects the best one based on a score.
 *
 * @details This kernel is designed to consolidate results, likely from a preceding calculation that 
 *          generated multiple potential solutions per event (e.g., different kinematic reconstructions). 
 *          It iterates through a list of candidate solutions (`i_sol`, `i_cmb`, `i_nu1`, `i_nu2`), grouped 
 *          by event using `evnt_dx` and potentially indexed by `cmx_dx`. For each event, it compares the 
 *          scores (`i_sol`) of its associated candidate solutions and identifies the solution with the 
 *          best score (e.g., lowest chi-squared, highest likelihood). The score, combination indices 
 *          (`i_cmb`), and corresponding neutrino parameters (`i_nu1`, `i_nu2`) of the winning solution 
 *          for each event are then written to the output tensors (`o_sol`, `o_cmb`, `o_nu1`, `o_nu2`). 
 *          Shared memory (sized by `size_x`) is likely used within each block to efficiently perform the 
 *          comparisons for the events assigned to that block.
 *
 * @tparam size_x Compile-time constant for shared memory allocation size per block.
 * @param[in] cmx_dx A 1D tensor potentially containing indices mapping candidate solutions to their 
 *                   original context or combination index. Shape: (lx).
 * @param[in] evnt_dx A 1D tensor mapping each candidate solution index (from 0 to lx-1) to its 
 *                    corresponding event index. This is crucial for grouping solutions by event. Shape: (lx).
 * @param[out] o_sol A 1D tensor to store the best score found for each event. Shape: (evnts).
 * @param[out] o_cmb A 2D tensor to store the combination indices associated with the best solution for 
 *                   each event. Shape: (evnts, num_indices_per_combination).
 * @param[out] o_nu1 A 2D tensor to store the first set of neutrino parameters corresponding to the best 
 *                   solution for each event. Shape: (evnts, num_params_nu1).
 * @param[out] o_nu2 A 2D tensor to store the second set of neutrino parameters corresponding to the best 
 *                   solution for each event. Shape: (evnts, num_params_nu2).
 * @param[in] i_cmb A 2D tensor containing the combination indices for each candidate solution. 
 *                  Shape: (lx, num_indices_per_combination).
 * @param[in] i_sol A 2D tensor containing the score for each candidate solution. The second dimension 
 *                  might be size 1, or contain multiple score components. Shape: (lx, num_score_components).
 * @param[in] i_nu1 A 3D tensor containing the first set of neutrino parameters for each candidate solution. 
 *                  The dimensions likely represent (candidate_index, parameter_component, parameter_set_index). 
 *                  Shape: (lx, num_params_nu1, num_sets_nu1). Or possibly (lx, num_params_nu1) if 3rd dim is trivial.
 * @param[in] i_nu2 A 3D tensor containing the second set of neutrino parameters for each candidate solution. 
 *                  Shape: (lx, num_params_nu2, num_sets_nu2). Or possibly (lx, num_params_nu2).
 * @param[in] lx The total number of candidate solutions across all events.
 * @param[in] evnts The total number of unique events for which solutions are being compared.
 */
template <size_t size_x>
__global__ void _compare_solx(
    torch::PackedTensorAccessor64<long  , 1, torch::RestrictPtrTraits> cmx_dx,
    torch::PackedTensorAccessor64<long  , 1, torch::RestrictPtrTraits> evnt_dx,

    torch::PackedTensorAccessor64<double, 1, torch::RestrictPtrTraits> o_sol,
    torch::PackedTensorAccessor64<long  , 2, torch::RestrictPtrTraits> o_cmb,
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> o_nu1,
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> o_nu2,

    torch::PackedTensorAccessor64<long  , 2, torch::RestrictPtrTraits> i_cmb,
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> i_sol,
    torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> i_nu1,
    torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> i_nu2, 
    const unsigned int lx, const unsigned int evnts
);

#endif // CU_NUSOL_UTILS_H
