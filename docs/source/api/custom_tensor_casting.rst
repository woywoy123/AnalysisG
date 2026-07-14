.. _custom-tensor-casting:

Tensor Casting Logic
====================

When building graphs or translating physics observables into machine learning formats, AnalysisG handles the casting from native C++ data structures into ``torch::Tensor`` objects dynamically. 

The ``to_tensor`` Interface
---------------------------
The primary casting mechanism is provided by the ``to_tensor`` template method, defined in ``graph_template.h``:

.. code-block:: cpp

    template <typename G, typename g>
    torch::Tensor to_tensor(std::vector<G> _data, at::ScalarType _op, g prim) {
        return build_tensor(&_data, _op, prim, this->op); 
    }

This function delegates to a backend tool (``build_tensor``) and accomplishes several critical steps:

1. **Memory Binding**: Translates ``std::vector<G>`` (where ``G`` can be ``bool``, ``float``, ``double``, ``int``, ``long``) into Torch's memory space.
2. **Scalar Types**: Explicitly assigns the underlying ``at::ScalarType`` (such as ``torch::kFloat``, ``torch::kLong``, etc.) to ensure the tensor operates correctly during neural network forward passes.
3. **Device Placement**: Captures device settings (CPU/CUDA) through ``this->op`` (``torch::TensorOptions``) so that the tensors can be seamlessly transferred to the GPU if required.

Type Deduplication
------------------
In ``src/AnalysisG/modules/graph/cxx/properties.cxx``, overloaded wrappers for all fundamental types exist:

- ``add_node_feature(std::vector<float> _data, std::string _name)`` maps to ``to_tensor(_data, torch::kFloat, float())``
- ``add_node_feature(std::vector<long> _data, std::string _name)`` maps to ``to_tensor(_data, torch::kLong, long())``

This approach enforces explicit type checking and minimizes implicit precision loss natively in C++ before handing the arrays over to the LibTorch backend.
