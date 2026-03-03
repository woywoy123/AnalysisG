Type Casting Utilities
======================

Template helpers for casting between C++ STL containers and PyTorch
tensors, and for merging containers across threads.

``cproperty<T, G>`` — Observable Property
------------------------------------------

``cproperty`` is a thin property-like wrapper used throughout the C++
class hierarchy.  It stores a value of type ``T`` and optionally calls a
user-supplied getter or setter callback (of type ``void(T*, G*)``) on each
read or write.  It models ``operator=(T)`` and implicit ``operator T()``,
so it behaves like a plain data member at call sites.

.. doxygenclass:: cproperty
   :project: AnalysisG

``merge_cast.h`` — Thread-safe Merge Helpers
---------------------------------------------

Recursive template functions for merging two data structures of the same
type (vectors, maps, or scalars) element-wise.  Used in multi-threaded
compilation to coalesce per-thread results into a single output.

- ``merge_data(out, p2)`` — append / overwrite ``*p2`` into ``*out``
- ``sum_data(out, p2)`` — add ``*p2`` to ``*out`` (numeric accumulation)
- ``reserve_count(inp, ix)`` — count elements for pre-allocation

.. doxygenfile:: merge_cast.h
   :project: AnalysisG

``tensor_cast.h`` — STL Container ↔ Tensor Conversion
-------------------------------------------------------

Recursive template system for converting arbitrarily nested
``std::vector<std::vector<…<T>…>>`` into flat ``torch::Tensor`` buffers
with automatic dimension tracking and null-padding for ragged tensors.

Key public functions:

- ``build_tensor(data, scalar_type, prim, opts)`` — convert a nested vector to a padded tensor
- ``tensor_to_vector(tensor, out)`` — round-trip back to a nested vector

.. doxygenfile:: tensor_cast.h
   :project: AnalysisG

``vector_cast.h`` — Variable Storage and Vector Casting
--------------------------------------------------------

Provides ``variable_t`` (a polymorphic type-erased buffer for a single
named ROOT leaf column) used by ``data_t`` and ``bsc_t``.

.. doxygenstruct:: variable_t
   :project: AnalysisG
   :members:
   :protected-members:
