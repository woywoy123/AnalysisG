.. cpp:namespace:: (global)

.. _typecasting_dox:

Type Casting and Tensor Utilities
=================================

This section describes definitions and implementations related to type casting, data processing, and tensor manipulation, primarily focusing on converting between different data structures like ``std::vector``, ``torch::Tensor``, and ROOT TTree branches. It defines the :cpp:struct:`variable_t` class for handling typed data and provides utility functions for tensor operations and data merging/summing/contracting.

variable_t Processing Overloads
-------------------------------

These methods handle associating different data types with the :cpp:struct:`variable_t` instance and potentially linking them to a TTree branch.

.. cpp:function:: void variable_t::process(std::vector<std::vector<float>>* data, std::string* varname, TTree* tr)

    Processes a 2D vector of floats.

    This overload handles data stored as a pointer to a ``std::vector<std::vector<float>>``. It calls the internal ``add_data`` method to associate this data with the corresponding internal member (``vv_f``) and potentially link it to a TTree branch.

    :param data: Pointer to the 2D vector of floats.
    :param varname: Pointer to the string representing the desired variable name in the TTree.
    :param tr: Pointer to the TTree object where the data branch should be created.

.. cpp:function:: void variable_t::process(std::vector<std::vector<double>>* data, std::string* varname, TTree* tr)

    Processes a 2D vector of doubles.

    This overload handles ``std::vector<std::vector<double>>`` data, associating it with the internal ``vv_d`` member via ``add_data``.

    :param data: Pointer to the 2D vector of doubles.
    :param varname: Pointer to the string representing the desired variable name.
    :param tr: Pointer to the TTree object.

.. cpp:function:: void variable_t::process(std::vector<std::vector<long>>* data, std::string* varname, TTree* tr)

    Processes a 2D vector of longs.

    This overload handles ``std::vector<std::vector<long>>`` data, associating it with the internal ``vv_l`` member via ``add_data``.

    :param data: Pointer to the 2D vector of longs.
    :param varname: Pointer to the string representing the desired variable name.
    :param tr: Pointer to the TTree object.

.. cpp:function:: void variable_t::process(std::vector<std::vector<int>>* data, std::string* varname, TTree* tr)

    Processes a 2D vector of ints.

    This overload handles ``std::vector<std::vector<int>>`` data, associating it with the internal ``vv_i`` member via ``add_data``.

    :param data: Pointer to the 2D vector of ints.
    :param varname: Pointer to the string representing the desired variable name.
    :param tr: Pointer to the TTree object.

.. cpp:function:: void variable_t::process(std::vector<std::vector<bool>>* data, std::string* varname, TTree* tr)

    Processes a 2D vector of bools.

    This overload handles ``std::vector<std::vector<bool>>`` data, associating it with the internal ``vv_b`` member via ``add_data``.

    :param data: Pointer to the 2D vector of bools.
    :param varname: Pointer to the string representing the desired variable name.
    :param tr: Pointer to the TTree object.

.. cpp:function:: void variable_t::process(std::vector<float>* data, std::string* varname, TTree* tr)

    Processes a 1D vector of floats.

    This overload handles ``std::vector<float>`` data, associating it with the internal ``v_f`` member via ``add_data``.

    :param data: Pointer to the 1D vector of floats.
    :param varname: Pointer to the string representing the desired variable name.
    :param tr: Pointer to the TTree object.

.. cpp:function:: void variable_t::process(std::vector<double>* data, std::string* varname, TTree* tr)

    Processes a 1D vector of doubles.

    This overload handles ``std::vector<double>`` data, associating it with the internal ``v_d`` member via ``add_data``.

    :param data: Pointer to the 1D vector of doubles.
    :param varname: Pointer to the string representing the desired variable name.
    :param tr: Pointer to the TTree object.

.. cpp:function:: void variable_t::process(std::vector<long>* data, std::string* varname, TTree* tr)

    Processes a 1D vector of longs.

    This overload handles ``std::vector<long>`` data, associating it with the internal ``v_l`` member via ``add_data``.

    :param data: Pointer to the 1D vector of longs.
    :param varname: Pointer to the string representing the desired variable name.
    :param tr: Pointer to the TTree object.

.. cpp:function:: void variable_t::process(std::vector<int>* data, std::string* varname, TTree* tr)

    Processes a 1D vector of ints.

    This overload handles ``std::vector<int>`` data, associating it with the internal ``v_i`` member via ``add_data``.

    :param data: Pointer to the 1D vector of ints.
    :param varname: Pointer to the string representing the desired variable name.
    :param tr: Pointer to the TTree object.

.. cpp:function:: void variable_t::process(std::vector<bool>* data, std::string* varname, TTree* tr)

    Processes a 1D vector of bools.

    This overload handles ``std::vector<bool>`` data, associating it with the internal ``v_b`` member via ``add_data``.

    :param data: Pointer to the 1D vector of bools.
    :param varname: Pointer to the string representing the desired variable name.
    :param tr: Pointer to the TTree object.

.. cpp:function:: void variable_t::process(float* data, std::string* varname, TTree* tr)

    Processes a single float value.

    This overload handles a pointer to a single ``float`` value, associating it with the internal ``f`` member via ``add_data``.

    :param data: Pointer to the float value.
    :param varname: Pointer to the string representing the desired variable name.
    :param tr: Pointer to the TTree object.

.. cpp:function:: void variable_t::process(double* data, std::string* varname, TTree* tr)

    Processes a single double value.

    This overload handles a pointer to a single ``double`` value, associating it with the internal ``d`` member via ``add_data``.

    :param data: Pointer to the double value.
    :param varname: Pointer to the string representing the desired variable name.
    :param tr: Pointer to the TTree object.

.. cpp:function:: void variable_t::process(long* data, std::string* varname, TTree* tr)

    Processes a single long value.

    This overload handles a pointer to a single ``long`` value, associating it with the internal ``l`` member via ``add_data``.

    :param data: Pointer to the long value.
    :param varname: Pointer to the string representing the desired variable name.
    :param tr: Pointer to the TTree object.

.. cpp:function:: void variable_t::process(int* data, std::string* varname, TTree* tr)

    Processes a single int value.

    This overload handles a pointer to a single ``int`` value, associating it with the internal ``i`` member via ``add_data``.

    :param data: Pointer to the int value.
    :param varname: Pointer to the string representing the desired variable name.
    :param tr: Pointer to the TTree object.

.. cpp:function:: void variable_t::process(bool* data, std::string* varname, TTree* tr)

    Processes a single bool value.

    This overload handles a pointer to a single ``bool`` value, associating it with the internal ``b`` member via ``add_data``.

    :param data: Pointer to the bool value.
    :param varname: Pointer to the string representing the desired variable name.
    :param tr: Pointer to the TTree object.

Tensor Utilities
----------------

.. cpp:function:: std::vector<signed long> tensor_size(torch::Tensor* inpt)

    Extracts the dimensions (shape) of a torch::Tensor.

    :param inpt: Pointer to the input torch::Tensor.
    :return: A ``std::vector<signed long>`` containing the size of each dimension of the tensor.

variable_t Class Definition
----------------------------

.. cpp:struct:: variable_t : public bsc_t

    A class designed to manage a variable of a specific type (determined at runtime) and interface it with ROOT TTrees and potentially torch::Tensors.

    Inherits from ``bsc_t`` (presumably a base class defined elsewhere). It holds data internally using pointers to various ``std::vector`` types or primitive types. It can determine the correct type based on an input ``torch::Tensor``, convert tensor data to the appropriate internal ``std::vector`` format, and create branches in a ``TTree`` linked to its internal data pointers.

    .. cpp:function:: variable_t()

        Default constructor.

    .. cpp:function:: variable_t(bool use_external)

        Constructor allowing specification of external data management.

        :param use_external: If true, indicates data might be managed externally.

    .. cpp:function:: virtual ~variable_t() override

        Destructor. Overrides the base class destructor. Ensures buffer flushing.

    .. cpp:function:: void create_meta(meta_t* mt)

        Creates and writes metadata to a "MetaData" TTree in the associated file.

        If a ``meta_t`` object is provided or already associated, it creates a temporary TTree named "MetaData", creates a branch named "MetaData" linked to the ``meta_t`` object, fills it once, writes it to the file, and then cleans up the temporary tree and pointers.

        :param mt: Pointer to a ``meta_t`` object containing metadata. If null, uses the internally stored ``mtx`` if available.

    .. cpp:function:: void build_switch(size_t s, torch::Tensor* tx)

        Determines and sets the internal data type enum based on tensor properties.

        This function inspects the number of dimensions (``s``) and the data type (``tx->dtype()``) of the input tensor to set the ``this->type`` member enum. It handles 3D, 2D, and 1D tensors for double, float, long, int, and bool types. If the combination of dimension and type is unrecognized, it prints an error message and aborts the program.

        :param s: The number of dimensions of the tensor.
        :param tx: Pointer to the input torch::Tensor.

        .. note::
            This function contains placeholders indicating where developers should add support for new tensor types/dimensions and corresponding enum values in ``modules/structs/base.h -> data_enum``.

    .. cpp:function:: void process(torch::Tensor* data, std::string* varname, TTree* tr)

        Processes data from a torch::Tensor, determines type, converts, and prepares for TTree branching.

        This is a central method for handling tensor data. It first determines the tensor's dimensions and, if the internal type is not yet set, calls :cpp:func:`build_switch` to determine and set the type based on the tensor's properties. It also handles the initial creation of metadata and association with the TTree. After potentially flushing buffers, it uses a switch statement based on the determined ``this->type`` to call the appropriate ``add_data`` template specialization, which converts the tensor data into the corresponding ``std::vector`` format and prepares it for TTree branching. Finally, it manages TTree branch caching.

        :param data: Pointer to the input torch::Tensor.
        :param varname: Pointer to the string representing the desired variable name. If the type is being set for the first time, this name is stored.
        :param tr: Pointer to the TTree object. If provided and no TTree is associated yet, it associates this TTree with the variable_t object.

        .. note::
            Contains a placeholder indicating where to add cases to the switch statement when supporting new data types.

    .. note::
        The various ``process`` overloads for ``std::vector`` and primitive types are documented :ref:`above <variable_t Processing Overloads>`.

    .. cpp:member:: std::string variable_name = ""

        The name assigned to this variable, used for TTree branching.

    .. cpp:member:: bool failed_branch = false

        Flag indicating if creating the TTree branch failed.

    Private Members
    ^^^^^^^^^^^^^^^

    .. cpp:member:: friend write_t

        Grant friendship to ``write_t``, allowing it access to private members.

    .. cpp:member:: bool use_external = false

        Flag indicating if external data management is used.

    .. cpp:member:: bool is_triggered = false

        Flag used internally, possibly related to TTree caching or event processing.

    .. cpp:member:: TBranch* tb = nullptr

        Pointer to the TBranch associated with this variable's data in the TTree.

    .. cpp:member:: TTree* tt = nullptr

        Pointer to the TTree where this variable's data is (or will be) stored.

    .. cpp:member:: meta_t* mtx = nullptr

        Pointer to associated metadata.

    .. cpp:function:: template <typename g, typename p> void add_data(g*& tx, torch::Tensor* data, std::vector<signed long>* s, p prim)

        Internal template method to add data from a torch::Tensor.

        Converts the tensor ``data`` to the appropriate std::vector structure ``tx`` using :cpp:func:`tensor_to_vector`. If the internal data pointer ``tx`` is null, it allocates memory for it. If the TBranch ``tb`` hasn't been created yet and a TTree ``tt`` is available, it creates the branch linking ``variable_name`` to the data pointer ``tx``. Sets ``failed_branch`` if branch creation fails.

        :tparam g: The potentially nested std::vector type (e.g., ``std::vector<std::vector<float>>``).
        :tparam p: The primitive type within the tensor/vector (e.g., ``float``).
        :param tx: Reference to the pointer holding the internal data structure.
        :param data: Pointer to the input torch::Tensor.
        :param s: Pointer to the vector of dimensions.
        :param prim: A primitive value of type ``p``, used for template deduction in :cpp:func:`tensor_to_vector`.

    .. cpp:function:: template <typename g> void add_data(g* var, g*& tx, std::string* name, TTree* tr = nullptr)

        Internal template method to add data from existing compatible types (vectors/primitives).

        Assigns the data from ``var`` to the internal data pointer ``tx``. If ``tx`` is null, it allocates memory and stores the provided ``name``. If the TBranch ``tb`` hasn't been created yet and a TTree ``tt`` (or the provided ``tr``) is available, it creates the branch linking ``variable_name`` to ``tx``. Sets ``failed_branch`` if creation fails. Sets ``is_triggered`` after successful branch creation.

        :tparam g: The data type (e.g., ``std::vector<float>``, ``int``).
        :param var: Pointer to the source data.
        :param tx: Reference to the pointer holding the internal data.
        :param name: Pointer to the variable name string.
        :param tr: Optional pointer to a TTree. If provided and ``this->tt`` is null, ``this->tt`` is set to ``tr``.


Merge, Sum, and Contract Utilities
==================================

These template functions provide utilities for combining or reshaping data structures like vectors and maps.

Merging Data
------------

.. cpp:function:: template <typename G> void merge_data(std::vector<G>* out, std::vector<G>* p2)

    Merges (appends) the contents of one vector into another.

    :tparam G: The type of elements stored in the vectors.
    :param out: Pointer to the destination vector where elements will be appended.
    :param p2: Pointer to the source vector whose elements will be appended.

.. cpp:function:: template <typename G> void merge_data(G* out, G* p2)

    Merges (assigns) the value of one primitive variable to another.

    :tparam G: The primitive data type.
    :param out: Pointer to the destination variable.
    :param p2: Pointer to the source variable.

.. cpp:function:: template <typename g, typename G> void merge_data(std::map<g, G>* out, std::map<g, G>* p2)

    Recursively merges the contents of one map into another.

    For each key-value pair in the source map (``p2``), it calls ``merge_data`` on the corresponding values in the destination map (``out``). If a key from ``p2`` doesn't exist in ``out``, it's implicitly created.

    :tparam g: The key type of the map.
    :tparam G: The value type of the map.
    :param out: Pointer to the destination map.
    :param p2: Pointer to the source map.

Summing Data
------------

.. cpp:function:: template <typename G> void sum_data(G* out, G* p2)

    Sums (adds) the value of one primitive variable to another.

    :tparam G: The primitive data type (must support ``+=`` operator).
    :param out: Pointer to the variable to which the value will be added.
    :param p2: Pointer to the variable whose value will be added.

.. cpp:function:: template <typename G> void sum_data(std::vector<G>* out, std::vector<G>* p2)

    Sums (appends) the contents of one vector into another.

    This function behaves identically to ``merge_data`` for vectors, appending elements.

    :tparam G: The type of elements stored in the vectors.
    :param out: Pointer to the destination vector.
    :param p2: Pointer to the source vector.

.. cpp:function:: template <typename g, typename G> void sum_data(std::map<g, G>* out, std::map<g, G>* p2)

    Recursively sums the contents of one map into another.

    For each key-value pair in the source map (``p2``), it calls ``sum_data`` on the corresponding values in the destination map (``out``). If a key from ``p2`` doesn't exist in ``out``, it's implicitly created (assuming the value type ``G`` is default-constructible).

    :tparam g: The key type of the map.
    :tparam G: The value type of the map (must support summation logic, e.g., ``+=`` or recursive ``sum_data``).
    :param out: Pointer to the destination map.
    :param p2: Pointer to the source map.

Contracting/Flattening Data
---------------------------

.. cpp:function:: template <typename g> void reserve_count(g* inp, long* ix)

    Counts a single element.

    Increments the counter pointed to by ``ix``. Base case for recursion.

    :tparam g: The type of the input element (unused).
    :param inp: Pointer to the input element (unused).
    :param ix: Pointer to the counter (long integer) to be incremented.

.. cpp:function:: template <typename g> void reserve_count(std::vector<g>* inp, long* ix)

    Recursively counts the total number of base elements within a nested vector structure.

    Iterates through the input vector and calls ``reserve_count`` for each element, effectively summing the counts from all nested levels down to the base elements.

    :tparam g: The type of elements in the vector (can be another vector).
    :param inp: Pointer to the input vector.
    :param ix: Pointer to the counter (long integer) to accumulate the total count.

.. cpp:function:: template <typename g> void contract_data(std::vector<g>* out, g* p2)

    Appends a single element to a vector.

    Base case for contracting nested structures. Pushes the value pointed to by ``p2`` onto the back of the ``out`` vector.

    :tparam g: The type of the element.
    :param out: Pointer to the destination vector.
    :param p2: Pointer to the element to be appended.

.. cpp:function:: template <typename g> void contract_data(std::vector<g>* out, std::vector<g>* p2)

    Flattens a vector of elements into a single vector.

    Iterates through the source vector ``p2`` and calls ``contract_data`` for each element, appending it to the ``out`` vector.

    :tparam g: The type of the elements.
    :param out: Pointer to the destination (flattened) vector.
    :param p2: Pointer to the source vector to be flattened.

.. cpp:function:: template <typename g> void contract_data(std::vector<g>* out, std::vector<std::vector<g>>* p2)

    Flattens a vector of vectors into a single vector, optimizing with reserve.

    First, it calculates the total number of elements in the nested structure using :cpp:func:`reserve_count`. Then, it reserves space in the output vector ``out`` for efficiency. Finally, it iterates through the outer vector ``p2`` and calls ``contract_data`` for each inner vector, effectively flattening the structure into ``out``.

    :tparam g: The type of the base elements.
    :param out: Pointer to the destination (flattened) vector.
    :param p2: Pointer to the source vector of vectors to be flattened.


Tensor Padding and Conversion Utilities
=======================================

These template functions handle padding ragged nested vectors and converting between nested vectors and flat primitive vectors or Tensors.

Padding and Standardization
---------------------------

.. cpp:function:: template <typename g> void scout_dim(g*, int*)

    Base case for scouting dimensions (does nothing for primitive types).

    :tparam g: Primitive data type.
    :param: Unused pointer.
    :param: Unused pointer to dimension count.

.. cpp:function:: template <typename G> void scout_dim(const std::vector<G>* vec, int* mx_dim)

    Recursively scouts the maximum dimension size at each level of a nested vector.

    Iterates through the vector ``vec``. For each element, it recursively calls ``scout_dim``. It keeps track of the maximum size encountered at the current level.

    :tparam G: The type of elements in the vector (potentially another vector).
    :param vec: Pointer to the vector to scout.
    :param mx_dim: Pointer to an integer storing the maximum dimension size found so far at the current nesting level. Updated if a larger size is found.

.. cpp:function:: template <typename g> void nulls(g* d, int*)

    Base case for setting null values (sets primitive to -1).

    :tparam g: Primitive data type.
    :param d: Pointer to the primitive value to set.
    :param: Unused pointer.

.. cpp:function:: template <typename g> void nulls(const std::vector<g>* d, int* mx_dim)

    Recursively pads a potentially ragged nested vector structure with default values.

    If the current vector ``d`` has fewer elements than ``*mx_dim``, it appends default-constructed elements (``{}``) and recursively calls ``nulls`` on them until the size matches ``*mx_dim``. This is intended to make ragged structures rectangular.

    :tparam g: The type of elements in the vector (potentially another vector).
    :param d: Pointer to the vector to pad.
    :param mx_dim: Pointer to the target dimension size for padding.

    .. note::
        This function modifies the vector pointed to by ``d`` by appending elements. It assumes ``g`` is default-constructible. The use of ``-1`` in the primitive ``nulls`` suggests this padding might be specific to numerical types where -1 indicates absence.

.. cpp:function:: template <typename g> bool standard(g*, int*)

    Base case for checking standardization (always true for primitives).

    :tparam g: Primitive data type.
    :param: Unused pointer.
    :param: Unused pointer.
    :return: Always returns true.

.. cpp:function:: template <typename g> bool standard(const std::vector<g>* vec, int* mx_dim)

    Checks if a nested vector structure is "standard" (rectangular) and pads it if not.

    Iterates through the vector ``vec``. If the vector is empty, it calls ``nulls`` to pad it. For each element, it recursively calls ``standard``. If any recursive call returns ``true`` (indicating a primitive level was reached) or if the structure was modified by padding, it calls ``nulls`` on the current vector ``vec`` to ensure rectangularity at this level.

    :tparam g: The type of elements in the vector (potentially another vector).
    :param vec: Pointer to the vector to check and potentially pad.
    :param mx_dim: Pointer to the target dimension size used for padding by ``nulls``.
    :return: Returns ``false`` generally, as its primary purpose is modification via ``nulls``. The return value seems less critical than the side effect of padding.

    .. note::
        This function modifies the vector pointed to by ``vec``.

Vector Flattening and Tensor Building
-------------------------------------

.. cpp:function:: template <typename G, typename g> void as_primitive(G* data, std::vector<g>* lin, std::vector<signed long>*, unsigned int)

    Base case for converting nested structures to a linear primitive vector.

    Appends the primitive value pointed to by ``data`` to the linear vector ``lin``.

    :tparam G: The (potentially nested) data type being processed.
    :tparam g: The primitive data type of the target linear vector.
    :param data: Pointer to the current primitive value.
    :param lin: Pointer to the accumulating linear vector of primitives.
    :param: Unused pointer to dimensions vector.
    :param: Unused depth counter.

.. cpp:function:: template <typename G, typename g> static void as_primitive(std::vector<G>* data, std::vector<g>* linear, std::vector<signed long>* dims, unsigned int depth = 0)

    Recursively converts a nested std::vector structure into a flat (linear) std::vector of primitives and determines its dimensions.

    Traverses the nested ``data`` vector. If the current depth matches the number of dimensions found so far, it means we are discovering a new dimension, so its size (current vector size) is added to the ``dims`` vector. It then recursively calls ``as_primitive`` on each element of the current vector, incrementing the depth. The base case (template specialization above) appends the primitive value to ``linear``.

    :tparam G: The type of elements in the potentially nested vector ``data``.
    :tparam g: The primitive type of the target ``linear`` vector.
    :param data: Pointer to the current level of the nested vector structure.
    :param linear: Pointer to the vector where flattened primitive values are accumulated.
    :param dims: Pointer to the vector where the dimensions of the structure are recorded.
    :param depth: The current recursion depth (starts at 0).

.. cpp:function:: template <typename G, typename g> static torch::Tensor build_tensor(std::vector<G>* _data, at::ScalarType _op, g, torch::TensorOptions* op)

    Builds a torch::Tensor from a potentially nested std::vector structure.

    This function orchestrates the conversion:
    1. Initializes variables for max dimension, the linear data vector, and dimensions.
    2. Calls :cpp:func:`scout_dim` to find the maximum size at each dimension level (for padding).
    3. Calls :cpp:func:`standard` to pad the input vector ``_data`` into a rectangular shape using ``nulls``.
    4. Calls :cpp:func:`as_primitive` to flatten the (now rectangular) ``_data`` into the ``linear`` vector and record the final ``dims``.
    5. Copies the data from the ``linear`` std::vector into a dynamically allocated C-style array ``d``.
    6. Handles a potential edge case where a 1D input results in only one dimension size; adds a second dimension of size 1.
    7. Creates a torch::Tensor ``ten`` using ``torch::from_blob``, pointing it to the C-style array ``d`` with the determined ``dims`` and the specified scalar type ``_op`` and tensor options ``op``.
    8. Clones the tensor to ensure it owns its memory.
    9. Deletes the temporary C-style array ``d``.
    10. Returns the created torch::Tensor.

    :tparam G: The potentially nested type of the input vector ``_data``.
    :tparam g: The primitive type contained within the nested structure ``_data``.
    :param _data: Pointer to the input nested std::vector. This vector might be modified by padding.
    :param _op: The target ``at::ScalarType`` (e.g., ``torch::kFloat32``) for the output tensor.
    :param: Unused primitive type parameter, likely for template deduction.
    :param op: Pointer to ``torch::TensorOptions`` to be used for tensor creation (e.g., device).
    :return: A ``torch::Tensor`` containing the data from ``_data``.


Vector Casting Utilities
========================

These functions handle chunking vectors and converting between Tensors and nested vectors.

.. cpp:function:: template <typename G> std::vector<std::vector<G>> chunking(std::vector<G>* v, int N)

    Splits a vector into chunks of a specified size.

    :tparam G: The type of elements in the vector.
    :param v: Pointer to the input vector to be chunked.
    :param N: The desired size of each chunk (the last chunk may be smaller).
    :return: A ``std::vector`` where each element is a ``std::vector<G>`` representing a chunk.

.. cpp:function:: template <typename g> void tensor_vector(std::vector<g>* trgt, std::vector<g>* chnks, std::vector<signed long>*, int)

    Base case for reconstructing nested vectors from a flat vector. Appends chunks directly.

    Appends all elements from the source chunk vector ``chnks`` to the target vector ``trgt``. This is the final step when the desired nested structure is reached.

    :tparam g: The primitive type of the elements.
    :param trgt: Pointer to the target vector where elements are appended.
    :param chnks: Pointer to the source vector containing a chunk of primitive elements.
    :param: Unused pointer to dimensions vector.
    :param: Unused dimension index.

.. cpp:function:: template <typename G, typename g> void tensor_vector(std::vector<G>* trgt, std::vector<g>* chnks, std::vector<signed long>* dims, int next_dim = 0)

    Recursively reconstructs a nested vector structure from a flat vector based on dimensions.

    Takes a flat chunk of data ``chnks`` and splits it into smaller chunks using :cpp:func:`chunking` based on the dimension size specified by ``(*dims)[next_dim]``. For each new smaller chunk, it creates a temporary nested vector ``tmp`` (of type ``G``) and recursively calls ``tensor_vector`` with the smaller chunk, the decremented dimension index (``next_dim-1``), and the temporary vector ``tmp`` as the target. Once the recursive call returns (having built the nested structure within ``tmp``), ``tmp`` is pushed back into the current target ``trgt``.

    :tparam G: The type of the vector elements at the current nesting level (e.g., ``std::vector<int>``).
    :tparam g: The primitive type of the flat data.
    :param trgt: Pointer to the target vector being constructed at the current nesting level.
    :param chnks: Pointer to the flat vector chunk for the current level.
    :param dims: Pointer to the vector containing the desired dimensions of the final nested structure.
    :param next_dim: The index into ``dims`` corresponding to the current nesting level's dimension size. Starts high and decrements.

.. cpp:function:: template <typename G, typename g> void tensor_to_vector(torch::Tensor* data, std::vector<G>* out, std::vector<signed long>* dims, g)

    Converts a torch::Tensor to a potentially nested std::vector structure.

    1. Creates an empty tensor ``cpux`` on the CPU with pinned memory, matching the input ``data`` tensor's size and type.
    2. Copies the data from the input ``data`` tensor (potentially on GPU) to the ``cpux`` tensor asynchronously.
    3. Synchronizes the CUDA device (if applicable) to ensure the copy is complete.
    4. Reshapes the ``cpux`` tensor into a 1D flat tensor.
    5. Creates a ``std::vector<g>`` named ``linear`` by directly constructing it from the raw data pointer (``cpux.data_ptr()``) of the flat ``cpux`` tensor. This avoids an extra data copy.
    6. Calls the recursive ``tensor_vector`` function to reconstruct the nested ``std::vector<G>`` structure ``out`` from the ``linear`` vector using the provided ``dims``.

    :tparam G: The potentially nested type of the output std::vector (e.g., ``std::vector<std::vector<float>>``).
    :tparam g: The primitive data type (e.g., ``float``) corresponding to the tensor's dtype.
    :param data: Pointer to the input torch::Tensor.
    :param out: Pointer to the output std::vector structure to be filled.
    :param dims: Pointer to the vector describing the dimensions of the tensor (and the target vector structure).
    :param: Unused primitive type parameter, likely for template deduction.

    .. note::
        Uses pinned memory for potentially faster CPU<->GPU transfers. Requires synchronization. Relies on the recursive ``tensor_vector`` function to rebuild the structure.

.. cpp:function:: template <typename g> void tensor_to_vector(torch::Tensor* data, std::vector<g>* out)

    Converts a torch::Tensor to a std::vector of primitives (implicitly flattens).

    This overload simplifies the conversion when the output is expected to be a flat ``std::vector<g>``. It first calls :cpp:func:`tensor_size` to get the tensor's dimensions and then calls the more detailed ``tensor_to_vector`` template function, passing the retrieved dimensions and a default-constructed ``g()`` for type deduction.

    :tparam g: The primitive data type of the tensor and the target vector.
    :param data: Pointer to the input torch::Tensor.
    :param out: Pointer to the output flat std::vector<g> to be filled.
