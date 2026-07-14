Mergecast Utilities
===================

The `mergecast` module (``src/AnalysisG/modules/typecasting/include/tools/merge_cast.h``) provides a collection of templated C++ functions for aggregating, summing, and flattening complex, nested data structures across threads. 

These utilities are heavily utilized inside the ``merge()`` routine of a ``Selection`` to safely collect thread-local `std::vector` and `std::map` results into global memory.

Core Functions
--------------

### 1. ``merge_data``

The ``merge_data`` family of templates handles concatenation and map merging.

- **Vector Concatenation**: Appends the contents of one vector onto another.
- **Map Merging**: Recursively merges two maps. If a key exists in both maps, ``merge_data`` is called recursively on the underlying values.

.. code-block:: cpp

    template <typename G>
    void merge_data(std::vector<G>* out, std::vector<G>* p2);

    template <typename g, typename G>
    void merge_data(std::map<g, G>* out, std::map<g, G>* p2);


### 2. ``sum_data``

The ``sum_data`` family of templates is similar to ``merge_data`` but utilizes the ``+=`` operator to numerically add scalar values. If called on vectors, it falls back to concatenation.

- **Scalar Addition**: Calls ``+=`` on the underlying types.
- **Map Accumulation**: Recursively sums values of matching keys.

.. code-block:: cpp

    template <typename G>
    void sum_data(G* out, G* p2); // (*out) += (*p2)

    template <typename g, typename G>
    void sum_data(std::map<g, G>* out, std::map<g, G>* p2);


### 3. ``contract_data``

The ``contract_data`` templates flatten nested vectors into a single 1D vector. This is highly optimized using ``reserve_count`` to pre-allocate memory for the flattened array before insertion, preventing expensive dynamic resizing.

.. code-block:: cpp

    // Flattens a 2D vector into a 1D vector
    template <typename g>
    void contract_data(std::vector<g>* out, std::vector<std::vector<g>>* p2);

### Memory Management

The module also provides a Cython utility for releasing memory dynamically:

.. code-block:: cpp

    template <typename g>
    void release_vector(std::vector<g>* ipt) { 
        ipt->shrink_to_fit(); 
    }
