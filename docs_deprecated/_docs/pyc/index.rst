API Reference
=============

The PyC API is organized into several modules:

.. toctree::
   :maxdepth: 2

   transform
   physics
   operators
   graph
   utils

Module Overview
---------------

transform
^^^^^^^^^

The ``transform`` module provides functions for coordinate transformations:

* Cartesian to Polar conversions
* Eta-Phi calculations
* Angular coordinates transformations
* Reference frame transformations

physics
^^^^^^^

The ``physics`` module contains fundamental physics calculations:

* Energy calculations
* Momentum operations
* Mass calculations
* Beta and gamma relativistic factors
* Transverse quantities (PT, ET)

operators
^^^^^^^^^

The ``operators`` module provides mathematical operators for physics quantities:

* Vector addition and subtraction
* Scalar multiplication
* Dot and cross products
* Magnitude calculations
* Angular separations

graph
^^^^^

The ``graph`` module contains utilities for graph operations in physics:

* Node feature extraction
* Edge feature calculations
* Graph construction utilities
* Graph transformation functions

utils
^^^^^

The ``utils`` module provides general utility functions:

* Tensor manipulation helpers
* Device management
* Type conversion utilities
* Random number generators















PyC Documentation
=================

PyC (Python CUDA) is a high-performance physics computation library built on top of PyTorch. It provides 
efficient implementations of common physics calculations for particle physics analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   api/index
   examples
   benchmarks



Introduction to PyC
Introduction to PyC
==================

Overview
--------

PyC (Python CUDA) is a high-performance physics computation library designed specifically for particle physics analysis.
It leverages the computational power of both CPUs and GPUs to perform complex physics calculations efficiently.

Key Features
-----------

*   **High Performance**: Optimized implementations for both CPU and CUDA
*   **Physics Focus**: Specialized functions for particle physics calculations
*   **Coordinate Transformations**: Efficient conversions between coordinate systems
*   **PyTorch Integration**: Seamless integration with PyTorch tensors
*   **Vector Operations**: Optimized vector calculations for physics use cases

Architecture
-----------

PyC is organized into several modules:

*   **transform**: Functions for coordinate transformations (Cartesian, polar, etc.)
*   **physics**: Core physics calculations (mass, energy, momentum, etc.)
*   **operators**: Mathematical operators for physics quantities
*   **graph**: Utilities for graph-based physics data representations
*   **utils**: Helper functions and utilities

The library provides both CPU-based implementations (tpyc) and GPU-accelerated versions (cupyc) of each function,
automatically selecting the appropriate one based on the input tensor's device placement.

Use Cases
--------

PyC is particularly useful for:

*   High-energy particle physics analysis
*   Processing large datasets of collision events
*   Feature engineering for machine learning in physics
*   Real-time physics calculations in data pipelines

Getting Started
-------------

To begin using PyC, see the :doc:`installation` guide and check out the :doc:`examples` for practical use cases.

Benchmarks
----------

PyC has been benchmarked against other common libraries to demonstrate its performance advantages.
Detailed benchmark results can be found in the :doc:`benchmarks` section.



Installation
============

Requirements
-----------

PyC requires the following dependencies:

* Python 3.7+
* PyTorch 1.8+
* CUDA Toolkit 11.0+ (for GPU acceleration)
* NumPy 1.19+
* AnalysisG core package

Installing PyC
-------------

PyC is included as part of the AnalysisG package. To install:

.. code-block:: bash

    git clone https://github.com/woywoy123/AnalysisG.git
    cd AnalysisG
    pip install -e .

pyc Function Examples
---------------------

The `pyc` interface provides access to optimized C++/CUDA functions for common operations, designed to work seamlessly with PyTorch tensors on both CPU and GPU. If CUDA is available and PyTorch tensors are on the GPU, `pyc` operations will automatically utilize CUDA acceleration where available.

You can inspect the available methods using `dir(pyc_instance)`. Below are examples demonstrating how to use some of these functions.

**Setup for Examples**

The following setup code imports necessary libraries, instantiates the `pyc` interface, determines the device (CPU or CUDA), and defines a helper function to create tensors.

.. code-block:: python

    import torch
    import random
    import math
    from AnalysisG.pyc import pyc # Import the pyc interface

    # Instantiate the interface
    pyc_instance = pyc()
    device = "cuda" if pyc_instance.cuda_available() and torch.cuda.is_available() else "cpu"
    print(f"--- Setup ---")
    print(f"Using device: {device}")
    print(f"pyc version: {pyc_instance.__version__}")
    print(f"Available pyc methods (partial): { [m for m in dir(pyc_instance) if not m.startswith('_')] }") # Show available methods

    def _make_tensor(shape, device, dtype=torch.float64, low=0.0, high=1.0):
        """Helper to create a tensor with random values."""
        return torch.rand(shape, dtype=dtype, device=device) * (high - low) + low

    # Define tolerance for floating point comparisons
    tolerance = 1e-6

    # --- PyTorch comparison functions (if needed) ---
    def torch_pmu_sum(pmu_tensor):
        # pmu_tensor shape: (N, 4) -> [px, py, pz, E]
        if pmu_tensor.shape[0] == 0:
            return torch.zeros(4, device=pmu_tensor.device, dtype=pmu_tensor.dtype)
        return pmu_tensor.sum(dim=0)

    def torch_delta_r(eta1, phi1, eta2, phi2):
        deta = eta1 - eta2
        dphi = phi1 - phi2
        # Ensure dphi is in [-pi, pi]
        dphi = torch.remainder(dphi + math.pi, 2 * math.pi) - math.pi
        return torch.sqrt(deta**2 + dphi**2)

    def torch_cartesian_to_pt_eta_phi_m(cartesian_tensor):
        # cartesian_tensor shape: (N, 4) -> [px, py, pz, E]
        px, py, pz, E = cartesian_tensor.unbind(-1)
        pt = torch.sqrt(px**2 + py**2)
        eta = torch.asinh(pz / pt)
        phi = torch.atan2(py, px)
        m_sq = E**2 - px**2 - py**2 - pz**2
        # Handle potential negative values due to precision
        m = torch.sqrt(torch.clamp(m_sq, min=0.0))
        # Handle cases where pt is zero (replace eta with 0 or +/- inf based on pz)
        eta = torch.where(pt == 0, torch.sign(pz) * float('inf'), eta)
        eta = torch.nan_to_num(eta, nan=0.0, posinf=float('inf'), neginf=float('-inf')) # Ensure NaNs from 0/0 become 0
        return torch.stack([pt, eta, phi, m], dim=-1)

    def torch_pt_eta_phi_m_to_cartesian(pmu_tensor):
        # pmu_tensor shape: (N, 4) -> [pt, eta, phi, m]
        pt, eta, phi, m = pmu_tensor.unbind(-1)
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        # Calculate E from m^2 = E^2 - p^2 => E = sqrt(m^2 + p^2)
        # p^2 = px^2 + py^2 + pz^2 = pt^2 + pz^2
        E = torch.sqrt(m**2 + pt**2 + pz**2)
        return torch.stack([px, py, pz, E], dim=-1)

**Example: Dot Product (`operators_dot` or `cuda_operators_dot`)**

Calculates the row-wise dot product between two matrices. This example checks for both a generic `operators_dot` and a CUDA-specific version.

.. code-block:: python

    print("\n--- Dot Product Example ---")
    dot_func_name = None
    if hasattr(pyc_instance, 'cuda_operators_dot') and device == "cuda":
        dot_func_name = 'cuda_operators_dot'
    elif hasattr(pyc_instance, 'operators_dot'):
        dot_func_name = 'operators_dot'

    if dot_func_name:
        print(f"Using pyc function: {dot_func_name}")
        rows, cols = 100, 50
        t1 = _make_tensor((rows, cols), device)
        t2 = _make_tensor((rows, cols), device)

        # Calculate dot product using pyc
        dot_pyc = getattr(pyc_instance, dot_func_name)(t1, t2)
        print(f"pyc {dot_func_name} result (first 10):")
        print(dot_pyc[:10].view(-1)) # Flatten for easier viewing

        # Calculate dot product using PyTorch for comparison
        dot_torch = (t1 * t2).sum(dim=-1)
        print("\nPyTorch dot product result (first 10):")
        print(dot_torch[:10])

        # Verify results are close
        diff = torch.abs(dot_pyc.view(-1) - dot_torch)
        print(f"\nResults match within tolerance: {torch.all(diff < tolerance)}")
    else:
        print("Skipping pyc dot product example: 'operators_dot' or 'cuda_operators_dot' not found or device mismatch.")

**Example: Sum of Four-Momenta (`operators_pmu_sum`)**

Calculates the sum of a list of four-momenta (px, py, pz, E).

.. code-block:: python

    print("\n--- Four-Momenta Sum Example ---")
    pmu_sum_func_name = 'operators_pmu_sum' # Assuming a generic name

    if hasattr(pyc_instance, pmu_sum_func_name):
        print(f"Using pyc function: {pmu_sum_func_name}")
        num_vectors = 1000
        # Create a tensor of shape (N, 4) representing N four-vectors
        pmu_vectors = _make_tensor((num_vectors, 4), device, low=-10, high=10)
        # Ensure Energy (E) is positive and large enough
        pmu_vectors[:, 3] = torch.sqrt(pmu_vectors[:, 0]**2 + pmu_vectors[:, 1]**2 + pmu_vectors[:, 2]**2 + _make_tensor((num_vectors,), device, low=1, high=5)**2)


        # Calculate sum using pyc
        sum_pyc = getattr(pyc_instance, pmu_sum_func_name)(pmu_vectors)
        print(f"pyc {pmu_sum_func_name} result:")
        print(sum_pyc)

        # Calculate sum using PyTorch for comparison
        sum_torch = torch_pmu_sum(pmu_vectors)
        print("\nPyTorch sum result:")
        print(sum_torch)

        # Verify results are close
        diff = torch.abs(sum_pyc - sum_torch)
        print(f"\nResults match within tolerance: {torch.all(diff < tolerance)}")
    else:
        print(f"Skipping pyc pmu sum example: '{pmu_sum_func_name}' not found.")

**Example: Delta R Calculation (`operators_delta_r`)**

Calculates the Delta R distance between pairs of particles given their eta and phi coordinates. Delta R = sqrt( (eta1-eta2)^2 + (phi1-phi2)^2 ), handling phi wrapping.

.. code-block:: python

    print("\n--- Delta R Example ---")
    delta_r_func_name = 'operators_delta_r' # Assuming a generic name

    if hasattr(pyc_instance, delta_r_func_name):
        print(f"Using pyc function: {delta_r_func_name}")
        num_pairs = 500
        # Create eta and phi tensors
        eta1 = _make_tensor((num_pairs,), device, low=-5, high=5)
        phi1 = _make_tensor((num_pairs,), device, low=-math.pi, high=math.pi)
        eta2 = _make_tensor((num_pairs,), device, low=-5, high=5)
        phi2 = _make_tensor((num_pairs,), device, low=-math.pi, high=math.pi)

        # Calculate Delta R using pyc
        delta_r_pyc = getattr(pyc_instance, delta_r_func_name)(eta1, phi1, eta2, phi2)
        print(f"pyc {delta_r_func_name} result (first 10):")
        print(delta_r_pyc[:10])

        # Calculate Delta R using PyTorch for comparison
        delta_r_torch = torch_delta_r(eta1, phi1, eta2, phi2)
        print("\nPyTorch Delta R result (first 10):")
        print(delta_r_torch[:10])

        # Verify results are close
        diff = torch.abs(delta_r_pyc - delta_r_torch)
        # Need slightly larger tolerance for complex functions like atan2, sqrt etc.
        delta_r_tolerance = 1e-5
        print(f"\nResults match within tolerance ({delta_r_tolerance}): {torch.all(diff < delta_r_tolerance)}")
    else:
        print(f"Skipping pyc Delta R example: '{delta_r_func_name}' not found.")

**Example: Cartesian to Pt/Eta/Phi/Mass (`operators_cartesian_to_pt_eta_phi_m`)**

Converts four-momenta from Cartesian coordinates (px, py, pz, E) to cylindrical coordinates (pt, eta, phi, mass).

.. code-block:: python

    print("\n--- Cartesian to PtEtaPhiM Example ---")
    cart_to_pmu_func_name = 'operators_cartesian_to_pt_eta_phi_m'

    if hasattr(pyc_instance, cart_to_pmu_func_name):
        print(f"Using pyc function: {cart_to_pmu_func_name}")
        num_vectors = 10
        # Create Cartesian vectors [px, py, pz, E]
        px = _make_tensor((num_vectors, 1), device, low=-100, high=100)
        py = _make_tensor((num_vectors, 1), device, low=-100, high=100)
        pz = _make_tensor((num_vectors, 1), device, low=-100, high=100)
        mass = _make_tensor((num_vectors, 1), device, low=0, high=10) # Ensure non-negative mass
        E = torch.sqrt(px**2 + py**2 + pz**2 + mass**2)
        cartesian_vectors = torch.cat([px, py, pz, E], dim=1)

        print("Input Cartesian Vectors (first 5):")
        print(cartesian_vectors[:5])

        # Convert using pyc
        pmu_pyc = getattr(pyc_instance, cart_to_pmu_func_name)(cartesian_vectors)
        print(f"\npyc {cart_to_pmu_func_name} result (first 5) [pt, eta, phi, m]:")
        print(pmu_pyc[:5])

        # Convert using PyTorch for comparison
        pmu_torch = torch_cartesian_to_pt_eta_phi_m(cartesian_vectors)
        print("\nPyTorch conversion result (first 5) [pt, eta, phi, m]:")
        print(pmu_torch[:5])

        # Verify results are close
        # Compare pt, phi, m directly. Eta can be inf, handle separately or use high tolerance.
        diff_pt = torch.abs(pmu_pyc[:, 0] - pmu_torch[:, 0])
        # Phi difference needs careful handling around +/- pi
        diff_phi = torch.abs(pmu_pyc[:, 2] - pmu_torch[:, 2])
        diff_phi = torch.min(diff_phi, 2 * math.pi - diff_phi) # Account for wrapping
        diff_m = torch.abs(pmu_pyc[:, 3] - pmu_torch[:, 3])
        # For eta, check non-infinite values
        valid_eta_mask = torch.isfinite(pmu_torch[:, 1]) & torch.isfinite(pmu_pyc[:, 1])
        diff_eta = torch.abs(pmu_pyc[valid_eta_mask, 1] - pmu_torch[valid_eta_mask, 1])

        conv_tolerance = 1e-5
        match = (torch.all(diff_pt < conv_tolerance) and
                 torch.all(diff_phi < conv_tolerance) and
                 torch.all(diff_m < conv_tolerance) and
                 torch.all(diff_eta < conv_tolerance))
        # Also check if infinite values match
        inf_eta_mask = ~valid_eta_mask
        inf_match = torch.all(torch.sign(pmu_pyc[inf_eta_mask, 1]) == torch.sign(pmu_torch[inf_eta_mask, 1]))

        print(f"\nResults match within tolerance ({conv_tolerance}): {match and inf_match}")
    else:
        print(f"Skipping pyc Cartesian to PtEtaPhiM example: '{cart_to_pmu_func_name}' not found.")


**Example: Pt/Eta/Phi/Mass to Cartesian (`operators_pt_eta_phi_m_to_cartesian`)**

Converts four-momenta from cylindrical coordinates (pt, eta, phi, mass) back to Cartesian coordinates (px, py, pz, E).

.. code-block:: python

    print("\n--- PtEtaPhiM to Cartesian Example ---")
    pmu_to_cart_func_name = 'operators_pt_eta_phi_m_to_cartesian'

    if hasattr(pyc_instance, pmu_to_cart_func_name):
        print(f"Using pyc function: {pmu_to_cart_func_name}")
        num_vectors = 10
        # Create PtEtaPhiM vectors [pt, eta, phi, m]
        pt = _make_tensor((num_vectors, 1), device, low=0, high=100) # pt >= 0
        eta = _make_tensor((num_vectors, 1), device, low=-5, high=5)
        phi = _make_tensor((num_vectors, 1), device, low=-math.pi, high=math.pi)
        m = _make_tensor((num_vectors, 1), device, low=0, high=10) # m >= 0
        pmu_vectors = torch.cat([pt, eta, phi, m], dim=1)

        print("Input PtEtaPhiM Vectors (first 5):")
        print(pmu_vectors[:5])

        # Convert using pyc
        cartesian_pyc = getattr(pyc_instance, pmu_to_cart_func_name)(pmu_vectors)
        print(f"\npyc {pmu_to_cart_func_name} result (first 5) [px, py, pz, E]:")
        print(cartesian_pyc[:5])

        # Convert using PyTorch for comparison
        cartesian_torch = torch_pt_eta_phi_m_to_cartesian(pmu_vectors)
        print("\nPyTorch conversion result (first 5) [px, py, pz, E]:")
        print(cartesian_torch[:5])

        # Verify results are close
        diff = torch.abs(cartesian_pyc - cartesian_torch)
        conv_tolerance = 1e-5
        print(f"\nResults match within tolerance ({conv_tolerance}): {torch.all(diff < conv_tolerance)}")

        # Optional: Verify conversion round trip (if previous example ran)
        if 'pmu_pyc' in locals() and cart_to_pmu_func_name == 'operators_cartesian_to_pt_eta_phi_m':
             print("\n--- Round Trip Verification ---")
             # Convert pyc's pmu result back to cartesian using pyc
             round_trip_cartesian = getattr(pyc_instance, pmu_to_cart_func_name)(pmu_pyc)
             # Compare with original cartesian input (from previous example)
             round_trip_diff = torch.abs(round_trip_cartesian - cartesian_vectors)
             print(f"Round trip match (Cartesian -> PMU -> Cartesian) within tolerance: {torch.all(round_trip_diff < conv_tolerance * 10)}") # Allow slightly larger tolerance for round trip

    else:
        print(f"Skipping pyc PtEtaPhiM to Cartesian example: '{pmu_to_cart_func_name}' not found.")


*Remember to consult the specific AnalysisG documentation or use `help(pyc_instance)` and `dir(pyc_instance)` in a Python session to get the exact list of available `pyc` functions and their parameters.*

Building from Source
------------------

To build PyC from source, which might be necessary for development or specific environments:

.. code-block:: bash

    git clone https://github.com/woywoy123/AnalysisG.git
    cd AnalysisG/pyc
      # Return to the root directory and install in editable mode
    pip install -e .






