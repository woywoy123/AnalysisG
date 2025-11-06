Typecasting Functions
---------------------

The purpose of this collection of functions is to simplify several tasks. 
One of those being merging data containers recursively, or summing data without having to write this explicitly.
Not only does it serve to simplify writing complex algorithms, it is also central to creating arbitrary PyTorch tensors.
The collection of functions is split across multiple header files, with each being importable with the **tools** class.


Merge Cast
^^^^^^^^^^

A header file used to merge vectors and maps recursively.

.. code-block:: C++

   #include <tools/merge_cast.h>

.. cpp:function:: template <typename G> void merge_data(std::vector<G>* out, std::vector<G>* in)
    
   A function used to append input vector and output vector recursively.

.. cpp:function:: template <typename G> void merge_data(std::map<std::string, G>* out, std::map<std::string, G>* in)

   A function used to append input and output maps recursively.
   If `G` is a std::vector, then the other merge_data function is called to merge its contents into a single vector.

.. cpp:function:: template <typename g, typename G> void merge_data(std::map<g, G>* out, std::map<g, G>* in)

   A double templated function used to merge any map data types.

.. cpp:function:: template <typename G> void sum_data(G* out, G* in)

   Adds input and output values, if G is a float, double, int, std::string or any other primitive type.

.. cpp:function:: template <typename G> void sum_data(std::vector<G>* out, std::vector<G>* in)

   Appends the input and output vectors recursively.

.. cpp:function:: template <typename g, typename G> void sum_data(std::map<g, G>* out, std::map<g, G>* in)

   Recursively aggregates the values of input and output maps.
   Depending on typename G, this could be summing primitives (e.g. float, int, ...) or appending (nested) vectors.


PyTorch Tensor Casting 
^^^^^^^^^^^^^^^^^^^^^^

The functions contained within this header are used to construct arbitrary tensors from n-dimensional vectors and assert dimensionally retention.
Although technically complex, the collection of functions are used to initially scout the maximal dimensionality of the input vector.
If any of the nested vectors do not conform the usual tensor convention, then additional padding is added to the vector, such that it can be interpreted as a multi-dimensional tensor.

.. code-block:: C++

   #include <tools/tensor_cast.h>


.. cpp:function:: template <typename g> void scout_dim(g* data, int* mx_dim)

   Prescans the in put data type for maximal dimensionality.

.. cpp:function:: template <typename g> void nulls(g* d, int* mx_dim)

   Adds padding to the input data if needed.

.. cpp:function:: template <typename g> bool standard(g* data, int* mx_dim)

   Checks if the input data is conforming to standard tensor dimensionality.

.. cpp:function:: template <typename G, typename g> void as_primitive(G* data, std::vector<g>* lin, std::vector<signed long>* dims, int depth)

   Converts the data into a primitive one dimensional vector.

.. cpp:function:: template <typename G>  void scout_dim(const std::vector<G>* vec, int* mx_dim)

   Entry point to scanning the dimensionality.

.. cpp:function:: template <typename g> void nulls(const std::vector<g>* d, int* mx_dim)

   Entry point to padding assertion.

.. cpp:function:: template <typename g> bool standard(const std::vector<g>* vec, int* mx_dim)

   Entry point to standardization method.

.. cpp:function:: template <typename G, typename g> static void as_primitive(std::vector<G>* data, std::vector<g>* linear, std::vector<signed long>* dims, int depth = 0)

   Entry point of linearization of the input data.

.. cpp:function:: template <typename G, typename g> static torch::Tensor build_tensor(std::vector<G>* _data, at::ScalarType _op, g prim, torch::TensorOptions* op)

   Constructs the tensor from the put data gien the scalar type and any tensor options.


Vector Casting
^^^^^^^^^^^^^^

A set of functions used to convert tensors to nested vectors and writing ROOT dictionaries during runtime.
The ROOT dictionaries are required to store std::vectors within ROOT files.


.. code-block:: C++

   #include <tools/vector_cast.h>

.. cpp:function:: std::vector<signed long> tensor_size(torch::Tensor* inpt)

   Returns the dimensionality of a given tensor.

.. cpp:function:: template <typename G> std::vector<std::vector<G>> chunking(std::vector<G>* v, int N)

   Chunks the input tensor back into its original nested vector represention.

.. cpp:function:: template <typename g> void tensor_vector(std::vector<g>* trgt, std::vector<g>* chnks, std::vector<signed long>* dims, int next_dim = 0)

   Converts the input tensor to its original vector representation.

.. cpp:function:: template <typename G, typename g> void tensor_vector(std::vector<G>* trgt, std::vector<g>* chnks, std::vector<signed long>* dims, int next_dim = 0)

   Converts the input tensor to its original vector representation.

.. cpp:function:: template <typename G, typename g> void tensor_to_vector(torch::Tensor* data, std::vector<G>* out, std::vector<signed long>* dims, g prim)

   Converts the input tensor to its original vector representation.

.. cpp:function:: void add_to_dict(std::vector<std::vector<float>>* dummy)

   A special fucntion used to generate the ROOT dictionaries during runtime.

.. cpp:function:: void add_to_dict(std::vector<std::vector<double>>* dummy)

   A special fucntion used to generate the ROOT dictionaries during runtime.

.. cpp:function:: void add_to_dict(std::vector<std::vector<long>>* dummy)

   A special fucntion used to generate the ROOT dictionaries during runtime.

.. cpp:function:: void add_to_dict(std::vector<std::vector<int>>* dummy)

   A special fucntion used to generate the ROOT dictionaries during runtime.

.. cpp:function:: void add_to_dict(std::vector<float>* dummy)

   A special fucntion used to generate the ROOT dictionaries during runtime.

.. cpp:function:: void add_to_dict(std::vector<double>* dummy)

   A special fucntion used to generate the ROOT dictionaries during runtime.

.. cpp:function:: void add_to_dict(std::vector<long>* dummy)

   A special fucntion used to generate the ROOT dictionaries during runtime.

.. cpp:function:: void add_to_dict(std::vector<int>* dummy)

   A special fucntion used to generate the ROOT dictionaries during runtime.

.. cpp:function:: void add_to_dict(std::vector<bool>* dummy)

   A special fucntion used to generate the ROOT dictionaries during runtime.








