CUDA API Reference (Actual Implementation)
===========================================

This document describes the actual CUDA implementations in AnalysisG, which use PyTorch tensor operations.

Overview
--------

The CUDA code in AnalysisG is designed to work with PyTorch tensors, not raw arrays. All kernels use:

* ``torch::Tensor`` for input/output
* ``torch::PackedTensorAccessor`` for kernel access
* ``AT_DISPATCH_FLOATING_TYPES`` for type templating
* Automatic CPU/GPU device handling

**Location**: ``src/AnalysisG/pyc/*/`` with ``.cu`` and ``.cuh`` files

Physics Calculations (pyc/physics)
-----------------------------------

**Location**: ``pyc/physics/physics.cu`` and ``pyc/physics/include/physics/physics.cuh``

The physics module provides kinematic calculations on four-momentum tensors.

Momentum Magnitude
~~~~~~~~~~~~~~~~~~

.. cpp:function:: torch::Tensor physics_::P2(torch::Tensor* pmc)
   
   Compute squared momentum magnitude P² = px² + py² + pz².
   
   :param pmc: Four-momentum tensor [N, 4] with columns [px, py, pz, e]
   :type pmc: torch::Tensor*
   :returns: P² tensor [N, 1]
   :rtype: torch::Tensor
   
   **CUDA Kernel**: ``_P2K<scalar_t, 128><<<blocks, threads>>>``
   
   **Usage**:
   
   .. code-block:: cpp
   
      torch::Tensor pmc = torch::rand({1000, 4}, torch::kCUDA);
      torch::Tensor p2 = physics_::P2(&pmc);  // [1000, 1]

.. cpp:function:: torch::Tensor physics_::P(torch::Tensor* pmc)
   
   Compute momentum magnitude P = √(px² + py² + pz²).
   
   :param pmc: Four-momentum tensor [N, 4]
   :returns: P tensor [N, 1]
   
   **CUDA Kernel**: ``_PK<scalar_t, 128><<<blocks, threads>>>``

Velocity (Beta)
~~~~~~~~~~~~~~~

.. cpp:function:: torch::Tensor physics_::Beta2(torch::Tensor* pmc)
   
   Compute β² = P²/E² (squared velocity in natural units).
   
   :param pmc: Four-momentum tensor [N, 4]
   :returns: β² tensor [N, 1]
   
   **CUDA Kernel**: ``_Beta2<scalar_t, 128><<<blocks, threads>>>``

.. cpp:function:: torch::Tensor physics_::Beta(torch::Tensor* pmc)
   
   Compute β = P/E (velocity in natural units).
   
   :param pmc: Four-momentum tensor [N, 4]
   :returns: β tensor [N, 1]

Invariant Mass
~~~~~~~~~~~~~~

.. cpp:function:: torch::Tensor physics_::M2(torch::Tensor* pmc)
   
   Compute squared invariant mass M² = E² - P².
   
   :param pmc: Four-momentum tensor [N, 4]
   :returns: M² tensor [N, 1]
   
   **CUDA Kernel**: ``_M2<scalar_t, 128><<<blocks, threads>>>``
   
   **Example**:
   
   .. code-block:: cpp
   
      // Compute mass of particle combinations
      torch::Tensor combined_pmc = particle1_pmc + particle2_pmc;
      torch::Tensor mass_squared = physics_::M2(&combined_pmc);
      torch::Tensor mass = torch::sqrt(mass_squared);  // MeV

.. cpp:function:: torch::Tensor physics_::M(torch::Tensor* pmc)
   
   Compute invariant mass M = √(E² - P²).
   
   :param pmc: Four-momentum tensor [N, 4]
   :returns: M tensor [N, 1]

Transverse Mass
~~~~~~~~~~~~~~~

.. cpp:function:: torch::Tensor physics_::Mt2(torch::Tensor* pmc)
   
   Compute squared transverse mass Mt² = Et² - pt².
   
   :param pmc: Four-momentum tensor [N, 4] or [N, 2] with [pz, e]
   :returns: Mt² tensor [N, 1]

.. cpp:function:: torch::Tensor physics_::Mt(torch::Tensor* pmc)
   
   Compute transverse mass Mt = √(Et² - pt²).
   
   :param pmc: Four-momentum tensor
   :returns: Mt tensor [N, 1]

Angular Quantities
~~~~~~~~~~~~~~~~~~

.. cpp:function:: torch::Tensor physics_::Theta(torch::Tensor* pmc)
   
   Compute polar angle θ = atan2(pt, pz).
   
   :param pmc: Four-momentum or momentum tensor [N, 3+]
   :returns: θ tensor [N, 1] in radians

.. cpp:function:: torch::Tensor physics_::DeltaR(torch::Tensor* pmu1, torch::Tensor* pmu2)
   
   Compute angular separation ΔR = √(Δη² + Δφ²).
   
   :param pmu1: First four-momentum tensor [N, 4]
   :param pmu2: Second four-momentum tensor [N, 4]
   :returns: ΔR tensor [N, 1]
   
   **Usage**:
   
   .. code-block:: cpp
   
      torch::Tensor dr = physics_::DeltaR(&lepton_pmc, &jet_pmc);
      // Apply DR < 0.4 cut
      torch::Tensor mask = dr < 0.4;

**Overloads**: All functions have overloads accepting separate px, py, pz, e tensors.

Mathematical Operators (pyc/operators)
---------------------------------------

**Location**: ``pyc/operators/operators.cu`` and ``pyc/operators/include/operators/operators.cuh``

Vector Operations
~~~~~~~~~~~~~~~~~

.. cpp:function:: torch::Tensor operators_::Dot(torch::Tensor* v1, torch::Tensor* v2)
   
   Compute dot product between vectors.
   
   :param v1: First vector tensor [N, M, K]
   :param v2: Second vector tensor [N, M, K]
   :returns: Dot product tensor [N, M, M]
   
   **CUDA Kernel**: ``_dot<scalar_t, 16, 4><<<blocks, threads>>>``
   **Threads**: dim3(16, 4, 4)

.. cpp:function:: torch::Tensor operators_::Cross(torch::Tensor* v1, torch::Tensor* v2)
   
   Compute cross product v1 × v2.
   
   :param v1: First vector tensor [N, M, K, 3]
   :param v2: Second vector tensor [N, M, 3]
   :returns: Cross product tensor [N, M, K, 3]
   
   **CUDA Kernel**: ``_cross<scalar_t><<<blocks, threads>>>``

Angular Operations
~~~~~~~~~~~~~~~~~~

.. cpp:function:: torch::Tensor operators_::CosTheta(torch::Tensor* v1, torch::Tensor* v2, unsigned int lm=0)
   
   Compute cos(θ) between vectors using v1·v2 / (|v1||v2|).
   
   :param v1: First vector [N, M]
   :param v2: Second vector [N, M]
   :param lm: Optional vector dimension limit
   :returns: cos(θ) tensor [N, 1]
   
   **Optimized Path**: For M < 3, uses PyTorch operations instead of CUDA kernel
   
   **CUDA Kernel**: ``_costheta<scalar_t><<<blocks, threads, shared_mem>>>``
   **Shared Memory**: ``sizeof(double) * M * 2`` bytes

.. cpp:function:: torch::Tensor operators_::SinTheta(torch::Tensor* v1, torch::Tensor* v2, unsigned int lm=0)
   
   Compute sin(θ) = √(1 - cos²(θ)).
   
   :param v1: First vector [N, M]
   :param v2: Second vector [N, M]
   :returns: sin(θ) tensor [N, 1]

Rotation Matrices
~~~~~~~~~~~~~~~~~

.. cpp:function:: torch::Tensor operators_::Rx(torch::Tensor* angle)
   
   Rotation matrix around x-axis.
   
   :param angle: Rotation angles [N]
   :returns: Rotation matrices [N, 3, 3]
   
   **CUDA Kernel**: ``_Rx<scalar_t><<<blocks, threads>>>``
   **Threads**: dim3(64, 3, 3)

.. cpp:function:: torch::Tensor operators_::Ry(torch::Tensor* angle)
   
   Rotation matrix around y-axis.
   
   :param angle: Rotation angles [N]
   :returns: Rotation matrices [N, 3, 3]

.. cpp:function:: torch::Tensor operators_::Rz(torch::Tensor* angle)
   
   Rotation matrix around z-axis.
   
   :param angle: Rotation angles [N]
   :returns: Rotation matrices [N, 3, 3]

Graph Operations (pyc/graph)
-----------------------------

**Location**: ``pyc/graph/pagerank.cu`` and ``pyc/graph/include/graph/pagerank.cuh``

PageRank Algorithm
~~~~~~~~~~~~~~~~~~

.. cpp:function:: std::map<std::string, torch::Tensor> graph_::page_rank(torch::Tensor* edge_index, torch::Tensor* edge_scores, double alpha, double threshold, double norm_low, long timeout, int num_cls)
   
   Compute PageRank scores for graph nodes.
   
   :param edge_index: Edge connectivity [2, num_edges] with [src, dst] nodes
   :type edge_index: torch::Tensor* (long)
   :param edge_scores: Edge weights [num_edges, 1]
   :type edge_scores: torch::Tensor* (float)
   :param alpha: Damping factor (typically 0.85)
   :param threshold: Convergence threshold
   :param norm_low: Normalization lower bound
   :param timeout: Maximum iterations
   :param num_cls: Number of clusters
   :returns: Map with keys "pagerank" (scores), "clusters" (assignments), "count" (cluster sizes)
   
   **CUDA Kernels**:
   
   * ``_get_max_node<128><<<blocks, threads>>>`` - Count outgoing edges per node
   * ``_get_remapping<<<blocks, threads>>>`` - Build node remapping
   * ``_page_rank<scalar_t, 128><<<blocks, threads>>>`` - Iterative PageRank computation
   
   **Algorithm**:
   
   1. Build adjacency structure
   2. Initialize PageRank scores uniformly
   3. Iteratively update: PR(v) = (1-α)/N + α·Σ(PR(u)/outdegree(u))
   4. Cluster nodes based on final scores
   
   **Example**:
   
   .. code-block:: cpp
   
      // Graph with 100 nodes, 500 edges
      torch::Tensor edges = torch::randint(0, 100, {2, 500}, torch::kLong);
      torch::Tensor scores = torch::rand({500, 1});
      
      auto result = graph_::page_rank(&edges, &scores, 
                                      0.85,    // alpha
                                      1e-6,    // threshold
                                      0.01,    // norm_low
                                      100,     // max iterations
                                      10);     // num clusters
      
      torch::Tensor pagerank = result["pagerank"];  // [100, 1]
      torch::Tensor clusters = result["clusters"];  // [100, 1]

.. cpp:function:: std::map<std::string, torch::Tensor> graph_::page_rank_reconstruction(torch::Tensor* edge_index, torch::Tensor* edge_scores, torch::Tensor* pmc, double alpha, double threshold, double norm_low, long timeout, int num_cls)
   
   PageRank with particle four-momentum for physics reconstruction.
   
   :param pmc: Particle four-momenta [num_nodes, 4]
   :returns: PageRank results plus reconstructed objects

Neutrino Reconstruction (pyc/nusol)
------------------------------------

**Location**: ``pyc/nusol/cuda/*.cu`` and ``pyc/nusol/include/nusol/*.cuh``

The nusol module implements multiple neutrino reconstruction algorithms on GPU.

Single Neutrino (W → lν)
~~~~~~~~~~~~~~~~~~~~~~~~~

**Files**: ``nu.cu``, ``nusol.cu``

Solves for neutrino momentum given:

* Lepton four-momentum
* Missing transverse momentum (MET)
* W boson mass constraint

**Method**: Solves quadratic equation for neutrino pz:

.. math::

   M_W^2 = (E_\ell + E_\nu)^2 - (\vec{p}_\ell + \vec{p}_\nu)^2

Results in 0, 1, or 2 solutions depending on discriminant.

Double Neutrino (ttbar → WbWb → lνblνb)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Files**: ``nunu.cu``, ``intersection.cu``

Solves for two neutrino momenta in dileptonic top pair events.

**Constraints**:

* Two W mass constraints
* Two top mass constraints
* MET constraint

**Method**: Numerical optimization to find intersection of constraint surfaces.

Matrix Operations
~~~~~~~~~~~~~~~~~

**File**: ``matrix.cu``

Matrix utilities for neutrino solving:

* Matrix inversion
* Determinant calculation
* Linear system solving

Coordinate Transforms (pyc/transform)
--------------------------------------

**Location**: ``pyc/transform/transform.cu``

The transform module handles coordinate system conversions.

.. cpp:function:: torch::Tensor transform_::PxPyPzE2PtEtaPhiE(torch::Tensor* pmc)
   
   Convert Cartesian to cylindrical coordinates.
   
   :param pmc: [px, py, pz, e] tensor [N, 4]
   :returns: [pt, eta, phi, e] tensor [N, 4]
   
   **Formulas**:
   
   * pt = √(px² + py²)
   * φ = atan2(py, px)
   * θ = atan2(pt, pz)
   * η = -ln(tan(θ/2))

.. cpp:function:: torch::Tensor transform_::PtEtaPhiE2PxPyPzE(torch::Tensor* pmc)
   
   Convert cylindrical to Cartesian coordinates.
   
   :param pmc: [pt, eta, phi, e] tensor [N, 4]
   :returns: [px, py, pz, e] tensor [N, 4]

Atomic Operations (pyc/cutils)
-------------------------------

**Location**: ``pyc/cutils/atomic.cu``

The cutils module provides atomic operations for CUDA, but the actual implementation is minimal (single include line).

The atomic operations are likely defined in headers:

* ``include/utils/atomic.cuh``
* Used by other kernels for thread-safe operations

Build Configuration
-------------------

CUDA Thread Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Typical patterns in AnalysisG:

.. code-block:: cpp

   // Physics operations: 128 threads per block
   #define phys_th 128
   const unsigned int thx = (dx >= phys_th) ? phys_th : dx;
   const dim3 threads = dim3(thx, 3);  // 3 for x,y,z components
   const dim3 blocks = blk_(dx, thx, 3, 3);  // Helper computes blocks
   
   // Operators: 16x4x4 threads
   #define op_thread 16
   const dim3 threads = dim3(op_thread, 4, 4);
   const dim3 blocks = blk_(dx, op_thread, 4, 4, 4, 4);
   
   // Graph operations: Large shared memory
   const dim3 threads = dim3(1, dy);
   unsigned int shared_mem_size = sizeof(double) * dy * 2;
   kernel<<<blocks, threads, shared_mem_size>>>(...);

Type Dispatching
~~~~~~~~~~~~~~~~

All kernels use PyTorch's type dispatching:

.. code-block:: cpp

   AT_DISPATCH_FLOATING_TYPES(tensor->scalar_type(), "kernel_name", [&]{
       _kernel<scalar_t><<<blocks, threads>>>(...);
   });

This generates specialized versions for float and double.

Device Placement
~~~~~~~~~~~~~~~~

Operations automatically use the device of input tensors:

.. code-block:: cpp

   torch::Tensor out = torch::zeros({N, M}, MakeOp(input));
   // out is on same device as input (CPU or CUDA)

Building
--------

CUDA code is compiled during package installation:

.. code-block:: bash

   pip install -e .

**Requirements**:

* CUDA Toolkit 11.0+
* PyTorch with CUDA support
* C++14 or later compiler

**CMake Configuration**: Each module has ``CMakeLists.txt`` specifying CUDA compilation.

See Also
--------

* :doc:`cpp_modules_api`: C++ modules documentation
* :doc:`../pyc/overview`: PyC package overview
* PyTorch CUDA documentation: https://pytorch.org/docs/stable/notes/cuda.html
