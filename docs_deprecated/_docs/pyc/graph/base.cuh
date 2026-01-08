/**
 * @brief CUDA kernel for predicting graph topology based on edge indices and predictions.
 *
 * @details This kernel function processes edge information and prediction data to generate
 *          pairs of nodes representing the predicted topology. It iterates through the
 *          prediction tensor and, for each prediction, identifies the corresponding edge
 *          in the `edge_index` tensor. The identified node pair from the edge is then
 *          stored in the `pairs` output tensor. The kernel is launched with a grid
 *          and block configuration suitable for parallel processing of the predictions.
 *
 * @tparam scalar_t The data type for tensor elements (although not directly used for computation
 *                  in this specific kernel, it's often part of a larger templated structure).
 *                  The primary data type used here is `long` for indices.
 *
 * @param pairs A 3-dimensional packed tensor accessor (`torch::PackedTensorAccessor64<long, 3>`)
 *              used as output. It will store the pairs of node indices representing the
 *              predicted edges. The dimensions are typically [prediction_index, 0/1 (for pair), feature_index (often 1)].
 *              This tensor is modified by the kernel.
 * @param edge_index A 2-dimensional constant packed tensor accessor (`const torch::PackedTensorAccessor64<long, 2>`)
 *                   representing the graph's edge index list. Typically, the shape is [2, num_edges],
 *                   where `edge_index[0]` contains source nodes and `edge_index[1]` contains destination nodes.
 *                   This tensor provides the mapping from edge indices to node pairs.
 * @param pred A 1-dimensional constant packed tensor accessor (`const torch::PackedTensorAccessor64<long, 1>`)
 *             containing the indices of the predicted edges within the `edge_index` tensor.
 *             The values in this tensor correspond to the second dimension index of `edge_index`.
 * @param dx_lx The total number of predictions to process, which corresponds to the length
 *              (size of the first dimension) of the `pred` tensor. This value determines the
 *              number of parallel threads needed or the range of the loop within the kernel.
 */
template <typename scalar_t>
__global__ void _prediction_topology(
                    torch::PackedTensorAccessor64<long, 3, torch::RestrictPtrTraits> pairs,
            const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> edge_index,
            const torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> pred,
            const unsigned int dx_lx
);

/**
 * @brief CUDA kernel for summing features along predicted edges.
 *
 * @details This kernel aggregates features based on the node pairs identified in the
 *          `_prediction_topology` kernel (or similar logic). For each pair of nodes
 *          stored in the `pairs` tensor, it retrieves the corresponding features from
 *          the input feature tensor `pmi` and accumulates them into the output tensor `pmu`.
 *          Atomic operations might be required within the kernel if multiple threads
 *          write to the same location in `pmu`, depending on the specific aggregation strategy.
 *          The kernel parallelizes the summation process over the predicted edges or nodes.
 *
 * @tparam scalar_t The data type of the features being summed (e.g., float, double).
 *                  This type determines the precision of the feature aggregation.
 *
 * @param pairs A 3-dimensional constant packed tensor accessor (`const torch::PackedTensorAccessor64<long, 3>`)
 *              containing the pairs of node indices for which features need to be summed.
 *              This tensor is typically the output of a topology prediction step.
 *              The dimensions might be [prediction_index, 0/1 (for pair), feature_index (often 1)].
 * @param pmu A 3-dimensional packed tensor accessor (`torch::PackedTensorAccessor64<scalar_t, 3>`)
 *            used as output to store the summed features for each predicted entity (e.g., edge or aggregated node).
 *            The dimensions should correspond to the structure defined by `pairs` and the feature dimension.
 *            This tensor is modified by the kernel.
 * @param pmi A 2-dimensional constant packed tensor accessor (`const torch::PackedTensorAccessor64<scalar_t, 2>`)
 *            containing the input node features. The dimensions are typically [num_nodes, num_features].
 *            This tensor provides the source feature data to be aggregated.
 * @param pred_lx The number of predictions or pairs stored in the `pairs` tensor (size of the first dimension).
 *                This influences the kernel launch configuration or loop bounds.
 * @param node_lx The total number of nodes in the graph (size of the first dimension of `pmi`).
 *                Used for indexing into the `pmi` tensor.
 * @param node_fx The number of features per node (size of the second dimension of `pmi`).
 *                Used for indexing and determining the size of feature vectors.
 */
template <typename scalar_t>
__global__ void _edge_summing(
                    torch::PackedTensorAccessor64<long    , 3, torch::RestrictPtrTraits> pairs,
                    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> pmu,
            const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmi,
            const unsigned int pred_lx, const unsigned int node_lx, const unsigned int node_fx
);

/**
 * @brief CUDA kernel to efficiently find unique elements within rows or columns of a cluster map.
 *
 * @details This kernel identifies unique values within specified dimensions of the input `cluster_map`.
 *          It processes the `cluster_map` in parallel. For each element (or potentially row/column depending
 *          on `dim_i`, `dim_j`, `dim_k`), it compares values to find unique entries according to the
 *          defined logic (e.g., finding the first occurrence). The results (e.g., indices or flags
 *          indicating uniqueness) are stored in the `out_map`. The exact uniqueness criteria depend
 *          on the kernel's internal implementation and how `dim_i`, `dim_j`, `dim_k` are used.
 *          This is often a preliminary step for aggregation operations like `_unique_sum`.
 *
 * @tparam scalar_t The data type for tensor elements (although the primary type used here is `long` for indices/maps).
 *
 * @param out_map A 2-dimensional packed tensor accessor (`torch::PackedTensorAccessor64<long, 2>`)
 *                used as output. It stores the result of the unique operation, which could be
 *                indices of unique elements, a map indicating uniqueness, or aggregated unique values.
 *                This tensor is modified by the kernel.
 * @param cluster_map A 2-dimensional constant packed tensor accessor (`const torch::PackedTensorAccessor64<long, 2>`)
 *                    representing the input map where unique elements need to be found. The interpretation
 *                    of its dimensions depends on the specific use case (e.g., [element_index, cluster_id]).
 * @param dim_i The size of the first dimension of the `cluster_map` being processed.
 *              Used for indexing and defining the processing range.
 * @param dim_j The size of the second dimension of the `cluster_map` being processed.
 *              Used for indexing and defining the processing range.
 * @param dim_k An additional dimension parameter, potentially used for comparison logic or
 *              defining the scope within which uniqueness is determined (e.g., comparing across
 *              a third logical dimension not explicitly represented in the tensor shape).
 */
template <typename scalar_t>
__global__ void _fast_unique(
                    torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> out_map,
            const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> cluster_map,
            const unsigned int dim_i, const unsigned int dim_j, const unsigned int dim_k
);

/**
 * @brief CUDA kernel to sum features based on unique cluster assignments.
 *
 * @details This kernel performs a segmented sum operation. It aggregates features from the `features`
 *          tensor based on the grouping defined by the `cluster_map`. For each unique cluster or group
 *          identified (potentially using the output of `_fast_unique` or similar logic implicitly
 *          within this kernel), it sums the corresponding features. Atomic operations are crucial here
 *          to handle concurrent writes to the same cluster's sum in the `out` tensor by different threads.
 *          The kernel parallelizes the summation process over the elements or features.
 *
 * @tparam scalar_t The data type of the features being summed (e.g., float, double).
 *                  This type determines the precision of the feature aggregation.
 *
 * @param out A 2-dimensional packed tensor accessor (`torch::PackedTensorAccessor64<scalar_t, 2>`)
 *            used as output. It stores the summed features for each unique cluster/group.
 *            The dimensions are typically [num_unique_clusters, num_features].
 *            This tensor is modified by the kernel, likely using atomic additions.
 * @param cluster_map A 2-dimensional constant packed tensor accessor (`const torch::PackedTensorAccessor64<long, 2>`)
 *                    mapping elements (e.g., nodes, pixels) to cluster IDs. The dimensions might be
 *                    [element_index, cluster_id] or similar, defining the grouping for summation.
 * @param features A 2-dimensional constant packed tensor accessor (`const torch::PackedTensorAccessor64<scalar_t, 2>`)
 *                 containing the features corresponding to the elements mapped in `cluster_map`.
 *                 The dimensions are typically [num_elements, num_features].
 * @param dim_i The size of the first dimension relevant to the `cluster_map` and `features` (e.g., number of elements).
 *              Used for indexing and defining the processing range.
 * @param dim_j The size of the second dimension relevant to the `cluster_map` (e.g., could relate to cluster ID range or structure).
 *              Used for indexing and defining the processing range.
 * @param dim_k The size of the feature dimension (number of features per element, size of the second dimension of `features`).
 *              Used for indexing features.
 */
template <typename scalar_t>
__global__ void _unique_sum(
                    torch::PackedTensorAccessor64