C Utilities (cutils)
====================

Low-level C++/CUDA utility functions used internally by all pyc kernels.
These are not part of the public API but are documented here for completeness.

CPU/C++ Utilities (``utils.h``)
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - ``clip(tensor, dim)``
     - Extract column ``dim`` from a 2-D tensor as a 1-D view
   * - ``format(vector<Tensor>*)``
     - Stack a list of tensors as columns (each reshaped to Nx1) into an NxK tensor
   * - ``format(vector<Tensor*>)``
     - Pointer-vector overload of ``format``
   * - ``MakeOp(tensor*)``
     - Create a ``TensorOptions`` matching the device and dtype of the input tensor
   * - ``changedev(tensor*)``
     - No-op in CPU build; moves tensor to the appropriate device in CUDA build
   * - ``changedev(dev, tensor*)``
     - Move tensor to named device string (e.g. ``"cuda:0"``); no-op in CPU build

.. doxygenfile:: utils.h
   :project: AnalysisG

CUDA Atomic Device Helpers (``atomic.cuh``)
--------------------------------------------

These ``__device__`` template functions are inlined into every CUDA kernel
that needs them.  They provide numerically stable variants of common
operations.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - ``_cofactor<T>(M, idy, idz)``
     - Compute the (idy, idz) cofactor of a 3×3 matrix ``M``
   * - ``_div(p)``
     - Safe reciprocal: returns ``1/p`` or ``0`` if ``p == 0``
   * - ``_p2(p)``
     - Squared value: ``(*p) × (*p)``
   * - ``_clp(p)``
     - Round-trip clamp: rounds to 10 decimal places
   * - ``_sqrt(p)``
     - Sign-preserving square root: returns ``-√|p|`` for negative inputs
   * - ``_cmp(xx, yy, xy)``
     - Computes ``xy / √(xx × yy)`` (used for cosine-like ratios)
   * - ``_arccos(sm, pz)``
     - ``acos(pz / √sm)`` with safe division

CUDA Thread/Block Geometry (``utils.cuh``)
-------------------------------------------

The ``blk_`` inline computes the 2-D CUDA launch grid:

.. code-block:: cuda

    dim3 blk = blk_(num_elements, threads_per_block);

This is used in every CUDA kernel launch to ensure full coverage of the
batch dimension.
