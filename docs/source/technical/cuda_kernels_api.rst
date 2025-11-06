CUDA Kernels API Reference
===========================

This section documents the CUDA implementations for GPU-accelerated operations in AnalysisG.

Overview
--------

The PyC package contains CUDA kernels for:

* Physics calculations (ΔR, invariant mass, etc.)
* Graph operations (PageRank, message passing)
* Neutrino reconstruction
* Mathematical operators
* Coordinate transformations
* Atomic operations

All CUDA code is in ``src/AnalysisG/pyc/*/cuda/`` and ``*.cu`` files.

CUDA Architecture
-----------------

AnalysisG CUDA code follows these patterns:

1. **Kernel Functions**: Marked with ``__global__`` for device execution
2. **Device Functions**: Marked with ``__device__`` for kernel-only calls
3. **Host Functions**: CPU interface that launches kernels
4. **Memory Management**: Efficient CPU ↔ GPU transfers
5. **Error Checking**: CUDA error handling throughout

Graph Operations (pyc/graph)
-----------------------------

PageRank CUDA Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``pyc/graph/cuda/pagerank.cu``

GPU-accelerated PageRank algorithm for graph analysis.

**Kernel Signature**:

.. code-block:: cuda

   __global__ void pagerank_kernel(
       const int* edge_index,     // Edge connectivity [2, num_edges]
       float* pagerank_scores,    // Output scores [num_nodes]
       const int num_nodes,
       const int num_edges,
       const float damping,       // Damping factor (typically 0.85)
       const int max_iterations
   );

**Usage Pattern**:

.. code-block:: cpp

   // Host code
   void pagerank_cuda(
       torch::Tensor edge_index,
       torch::Tensor scores,
       float damping = 0.85f,
       int max_iters = 100
   ) {
       int num_nodes = scores.size(0);
       int num_edges = edge_index.size(1);
       
       // Launch kernel
       int block_size = 256;
       int num_blocks = (num_nodes + block_size - 1) / block_size;
       
       pagerank_kernel<<<num_blocks, block_size>>>(
           edge_index.data_ptr<int>(),
           scores.data_ptr<float>(),
           num_nodes, num_edges, damping, max_iters
       );
   }

**Performance**: Processes millions of nodes/edges efficiently on GPU.

Graph Construction
~~~~~~~~~~~~~~~~~~

**Location**: ``pyc/graph/cuda/graph.cu``

Builds graph structures from particle data.

**Key Kernels**:

.. code-block:: cuda

   // Build edge list based on distance threshold
   __global__ void build_edges_kernel(
       const float* positions,     // [num_particles, 3]
       int* edge_index,           // Output [2, max_edges]
       int* num_edges,            // Output edge count
       const int num_particles,
       const float threshold      // Max distance for edge
   );
   
   // Compute edge features (ΔR, etc.)
   __global__ void edge_features_kernel(
       const float* node_features,  // [num_nodes, feat_dim]
       const int* edge_index,
       float* edge_features,        // Output
       const int num_edges
   );

Physics Calculations (pyc/physics)
-----------------------------------

DeltaR Calculation
~~~~~~~~~~~~~~~~~~

**Location**: ``pyc/physics/cuda/physics.cu``

Computes angular separation between particles.

**Kernel**:

.. code-block:: cuda

   __device__ float compute_deltaR(
       float eta1, float phi1,
       float eta2, float phi2
   ) {
       float deta = eta1 - eta2;
       float dphi = phi1 - phi2;
       
       // Wrap phi to [-π, π]
       while (dphi > M_PI) dphi -= 2*M_PI;
       while (dphi < -M_PI) dphi += 2*M_PI;
       
       return sqrtf(deta*deta + dphi*dphi);
   }
   
   __global__ void deltaR_matrix_kernel(
       const float* eta,          // [N]
       const float* phi,          // [N]
       float* dr_matrix,          // Output [N, N]
       const int N
   ) {
       int i = blockIdx.x * blockDim.x + threadIdx.x;
       int j = blockIdx.y * blockDim.y + threadIdx.y;
       
       if (i < N && j < N) {
           dr_matrix[i*N + j] = compute_deltaR(
               eta[i], phi[i], eta[j], phi[j]
           );
       }
   }

**Usage**: Computes all-pairs ΔR matrix for N particles in O(N²) but parallelized on GPU.

Invariant Mass
~~~~~~~~~~~~~~

**Kernel**:

.. code-block:: cuda

   __global__ void invariant_mass_kernel(
       const float* px,           // [N]
       const float* py,
       const float* pz,
       const float* e,
       const int* combinations,   // [M, K] - which particles to combine
       float* masses,             // Output [M]
       const int M,               // Number of combinations
       const int K                // Particles per combination
   ) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx >= M) return;
       
       // Sum four-momentum
       float total_px = 0, total_py = 0, total_pz = 0, total_e = 0;
       for (int k = 0; k < K; k++) {
           int particle_idx = combinations[idx * K + k];
           total_px += px[particle_idx];
           total_py += py[particle_idx];
           total_pz += pz[particle_idx];
           total_e += e[particle_idx];
       }
       
       // Compute invariant mass
       float p2 = total_px*total_px + total_py*total_py + total_pz*total_pz;
       masses[idx] = sqrtf(total_e*total_e - p2);
   }

Neutrino Reconstruction (pyc/nusol)
------------------------------------

**Location**: ``pyc/nusol/cuda/``

GPU-accelerated neutrino reconstruction using multiple algorithms.

Analytical W → lν Solver
~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``analytical_w.cu``

Solves for neutrino pz given W boson mass constraint.

**Kernel**:

.. code-block:: cuda

   __global__ void solve_neutrino_pz_kernel(
       const float* lepton_px,     // [N]
       const float* lepton_py,
       const float* lepton_pz,
       const float* lepton_e,
       const float* met_px,        // Missing ET x
       const float* met_py,        // Missing ET y
       float* nu_pz_solutions,     // Output [N, 2] (two solutions)
       int* num_solutions,         // Output [N] (0, 1, or 2)
       const float W_mass,
       const int N
   ) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx >= N) return;
       
       // Solve quadratic equation for neutrino pz
       // (details omitted for brevity)
       
       float a = /* ... */;
       float b = /* ... */;
       float c = /* ... */;
       
       float discriminant = b*b - 4*a*c;
       
       if (discriminant < 0) {
           num_solutions[idx] = 0;
       } else if (discriminant == 0) {
           nu_pz_solutions[idx*2] = -b / (2*a);
           num_solutions[idx] = 1;
       } else {
           float sqrt_d = sqrtf(discriminant);
           nu_pz_solutions[idx*2] = (-b + sqrt_d) / (2*a);
           nu_pz_solutions[idx*2 + 1] = (-b - sqrt_d) / (2*a);
           num_solutions[idx] = 2;
       }
   }

**Performance**: Solves thousands of neutrino reconstruction problems in parallel.

Numerical Optimization
~~~~~~~~~~~~~~~~~~~~~~

**File**: ``numerical_optimizer.cu``

Chi-squared minimization for top → Wb → lνb reconstruction.

**Key Features**:

* Parallel gradient descent
* Constraint handling
* Multiple initial conditions
* Best solution selection

Mathematical Operators (pyc/operators)
---------------------------------------

**Location**: ``pyc/operators/cuda/operators.cu``

CUDA implementations of common mathematical operations.

Matrix Operations
~~~~~~~~~~~~~~~~~

.. code-block:: cuda

   __global__ void matrix_multiply_kernel(
       const float* A,    // [M, K]
       const float* B,    // [K, N]  
       float* C,          // Output [M, N]
       int M, int K, int N
   );
   
   __global__ void transpose_kernel(
       const float* input,   // [M, N]
       float* output,        // [N, M]
       int M, int N
   );

Element-wise Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cuda

   __global__ void relu_kernel(float* data, int N);
   __global__ void sigmoid_kernel(float* data, int N);
   __global__ void softmax_kernel(float* data, int N, int dim);

Coordinate Transforms (pyc/transform)
--------------------------------------

**Location**: ``pyc/transform/cuda/transform.cu``

Coordinate system transformations.

Cartesian ↔ Spherical
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cuda

   __global__ void cartesian_to_spherical_kernel(
       const float* px,
       const float* py,
       const float* pz,
       float* pt,    // Output
       float* eta,   // Output
       float* phi,   // Output
       int N
   ) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx >= N) return;
       
       float px_val = px[idx];
       float py_val = py[idx];
       float pz_val = pz[idx];
       
       // pt = sqrt(px² + py²)
       pt[idx] = sqrtf(px_val*px_val + py_val*py_val);
       
       // phi = atan2(py, px)
       phi[idx] = atan2f(py_val, px_val);
       
       // eta = -ln(tan(θ/2)) where θ = atan2(pt, pz)
       float theta = atan2f(pt[idx], pz_val);
       eta[idx] = -logf(tanf(theta / 2.0f));
   }

Lorentz Boosts
~~~~~~~~~~~~~~

.. code-block:: cuda

   __global__ void lorentz_boost_kernel(
       const float* px_in, const float* py_in,
       const float* pz_in, const float* e_in,
       float* px_out, float* py_out,
       float* pz_out, float* e_out,
       float beta_x, float beta_y, float beta_z,
       int N
   );

Atomic Operations (pyc/cutils)
-------------------------------

**Location**: ``pyc/cutils/cuda/atomic.cu``

Thread-safe atomic operations for CUDA.

Custom Atomics
~~~~~~~~~~~~~~

.. code-block:: cuda

   __device__ float atomicMaxFloat(float* address, float val) {
       int* address_as_int = (int*)address;
       int old = *address_as_int, assumed;
       
       do {
           assumed = old;
           old = atomicCAS(address_as_int, assumed,
               __float_as_int(fmaxf(val, __int_as_float(assumed))));
       } while (assumed != old);
       
       return __int_as_float(old);
   }
   
   __device__ double atomicAddDouble(double* address, double val) {
       unsigned long long int* address_as_ull =
           (unsigned long long int*)address;
       unsigned long long int old = *address_as_ull, assumed;
       
       do {
           assumed = old;
           old = atomicCAS(address_as_ull, assumed,
               __double_as_longlong(val + __longlong_as_double(assumed)));
       } while (assumed != old);
       
       return __longlong_as_double(old);
   }

Performance Optimization
------------------------

Thread Configuration
~~~~~~~~~~~~~~~~~~~~

Typical kernel launch patterns:

.. code-block:: cpp

   // 1D grid
   int block_size = 256;  // Threads per block
   int grid_size = (N + block_size - 1) / block_size;
   kernel<<<grid_size, block_size>>>(args);
   
   // 2D grid (for matrices)
   dim3 block(16, 16);
   dim3 grid((M + 15) / 16, (N + 15) / 16);
   kernel<<<grid, block>>>(args);

Memory Patterns
~~~~~~~~~~~~~~~

**Coalesced Access**: Ensure adjacent threads access adjacent memory:

.. code-block:: cuda

   // Good: Coalesced
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   float val = data[idx];
   
   // Bad: Strided access
   int idx = threadIdx.x * blockDim.x + blockIdx.x;
   float val = data[idx];

**Shared Memory**: Use for data reuse within block:

.. code-block:: cuda

   __global__ void kernel_with_shared() {
       __shared__ float cache[256];
       
       int tid = threadIdx.x;
       cache[tid] = global_data[blockIdx.x * blockDim.x + tid];
       __syncthreads();
       
       // Use cache[] instead of global memory
   }

Error Handling
--------------

All CUDA calls should check for errors:

.. code-block:: cpp

   #define CUDA_CHECK(call) { \
       cudaError_t err = call; \
       if (err != cudaSuccess) { \
           fprintf(stderr, "CUDA error in %s:%d: %s\n", \
               __FILE__, __LINE__, cudaGetErrorString(err)); \
           exit(1); \
       } \
   }
   
   // Usage
   CUDA_CHECK(cudaMalloc(&d_data, size));
   kernel<<<grid, block>>>(args);
   CUDA_CHECK(cudaDeviceSynchronize());

Building CUDA Code
------------------

CUDA code is compiled automatically during installation if CUDA is available:

.. code-block:: bash

   # CUDA must be in PATH
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   
   # Install (will compile CUDA if available)
   pip install -e .

**Requirements**:

* CUDA Toolkit 11.0 or later
* Compatible GPU (Compute Capability 6.0+)
* nvcc compiler

See Also
--------

* :doc:`cpp_modules_api`: C++ modules documentation
* :doc:`../pyc/overview`: PyC package overview
* :doc:`overview`: Technical overview
