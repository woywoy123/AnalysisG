Type Casting Utilities
======================

Template helpers for casting between C++ STL containers and PyTorch
tensors, and for merging containers across threads.

.. doxygenfile:: merge_cast.h
   :project: AnalysisG

.. doxygenfile:: tensor_cast.h
   :project: AnalysisG

``vector_cast.h`` — Variable Storage and Vector Casting
--------------------------------------------------------

.. doxygenstruct:: variable_t
   :project: AnalysisG
   :members:
   :protected-members:

.. doxygenstruct:: write_t
   :project: AnalysisG
   :members:
