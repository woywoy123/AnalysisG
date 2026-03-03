Graph Template
==============

This module provides two closely related types:

* **``graph_t``** — a plain struct that holds pre-compiled tensor data for one
  graph (nodes, edges, truth labels, edge-index, batch-index, …).  It is the
  object passed to ``model_template::forward`` during training/inference.
* **``graph_template``** — the abstract base class for all user-defined graph
  types.  Users override ``CompileEvent()`` to call the feature-registration
  helpers, and optionally override ``PreSelection()`` to filter events.

Struct: ``graph_t``
-------------------

**Header:** ``<templates/graph_template.h>``

``graph_t`` is populated by the framework; user code only *reads* from it inside
``model_template::forward``.

Public Getter Templates (call from ``model_template::forward``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All getters take a model pointer (``this``) to resolve the device index.

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Signature
     - Description
   * - ``template<g> torch::Tensor* get_data_graph(std::string name, g* mdl)``
     - Returns the named graph-level *data* feature tensor.
   * - ``template<g> torch::Tensor* get_data_node(std::string name, g* mdl)``
     - Returns the named node-level *data* feature tensor (shape ``[N_nodes, 1]``).
   * - ``template<g> torch::Tensor* get_data_edge(std::string name, g* mdl)``
     - Returns the named edge-level *data* feature tensor (shape ``[N_edges, 1]``).
   * - ``template<g> torch::Tensor* get_truth_graph(std::string name, g* mdl)``
     - Returns the named graph-level *truth* label tensor.
   * - ``template<g> torch::Tensor* get_truth_node(std::string name, g* mdl)``
     - Returns the named node-level *truth* label tensor.
   * - ``template<g> torch::Tensor* get_truth_edge(std::string name, g* mdl)``
     - Returns the named edge-level *truth* label tensor.
   * - ``template<g> torch::Tensor* get_edge_index(g* mdl)``
     - Returns the ``[2, N_edges]`` COO edge-index tensor.
   * - ``template<g> torch::Tensor* get_event_weight(g* mdl)``
     - Returns the scalar event-weight tensor.
   * - ``template<g> torch::Tensor* get_batch_index(g* mdl)``
     - Returns the batch-index vector (non-null only during batched inference).
   * - ``template<g> torch::Tensor* get_batched_events(g* mdl)``
     - Returns a tensor of event indices in the current batch.

Public Data Fields
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Type
     - Description
   * - ``num_nodes``
     - ``int``
     - Number of nodes in this graph.
   * - ``event_index``
     - ``long``
     - Sequential event index from the source ROOT file.
   * - ``event_weight``
     - ``double``
     - Monte Carlo event weight.  Default ``1``.
   * - ``preselection``
     - ``bool``
     - Whether this graph passed the ``PreSelection()`` filter.
   * - ``hash``
     - ``std::string*``
     - Pointer to the event hash string.
   * - ``filename``
     - ``std::string*``
     - Pointer to the source ROOT file path.
   * - ``graph_name``
     - ``std::string*``
     - Pointer to the graph-class name string.
   * - ``device``
     - ``c10::DeviceType``
     - Current device (CPU or CUDA).
   * - ``batched_events``
     - ``std::vector<long>``
     - Event indices in the batch (non-empty during batched inference).
   * - ``in_use``
     - ``int``
     - Reference-count flag used by the framework memory manager.

Class: ``graph_template``
--------------------------

**Header:** ``<templates/graph_template.h>``

**Inheritance:** ``tools``

Properties
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 20 62

   * - Property
     - Type
     - Description
   * - ``index``
     - ``long``
     - Graph sequential index.  Read-only.
   * - ``weight``
     - ``double``
     - Event weight.  Read-only.
   * - ``preselection``
     - ``bool``
     - Whether this graph passed ``PreSelection()``.
   * - ``hash``
     - ``std::string``
     - 18-character event hash.  Read-only.
   * - ``tree``
     - ``std::string``
     - ROOT tree name.  Read-only.
   * - ``name``
     - ``std::string``
     - Graph-class name.  Writable.

Public Fields
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Field
     - Type
     - Description
   * - ``threadIdx``
     - ``int``
     - Worker-thread index (set by the framework).  Default ``-1``.
   * - ``filename``
     - ``std::string``
     - Source ROOT file path.
   * - ``meta_data``
     - ``meta*``
     - Pointer to the dataset metadata object.

Public Methods
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Signature
     - Description
   * - ``virtual graph_template* clone()``
     - Override in subclasses to return a heap-allocated copy.
   * - ``virtual void CompileEvent()``
     - **Primary override point.** Call ``define_particle_nodes``,
       ``define_topology``, and all ``add_*_feature`` methods here.
   * - ``virtual bool PreSelection()``
     - Return ``false`` to discard this event before feature compilation.
       Default returns ``true``.
   * - ``void define_particle_nodes(std::vector<particle_template*>* prt)``
     - Registers *prt* as the node collection for this graph.  Must be called
       before any ``add_node_*`` or ``add_edge_*`` feature calls.
   * - ``void define_topology(std::function<bool(particle_template*, particle_template*)> fx)``
     - Provides a binary predicate ``fx(pi, pj)`` that determines whether to
       include the directed edge i→j.  If not called, the default full
       topology (all pairs, including self-loops) is used.
   * - ``template<G,O,X> void add_graph_truth_feature(O* ev, X fx, std::string name)``
     - Reads event-level property *G* via getter *fx* on object *ev* and stores
       as a graph-level truth feature.
   * - ``template<G,O,X> void add_graph_data_feature(O* ev, X fx, std::string name)``
     - Like ``add_graph_truth_feature`` but stores as graph-level *data*.
   * - ``template<G,O,X> void add_node_truth_feature(X fx, std::string name)``
     - Evaluates *fx* on every registered node particle (cast to *O*) and
       stores as a node-level truth feature.
   * - ``template<G,O,X> void add_node_data_feature(X fx, std::string name)``
     - Like ``add_node_truth_feature`` but stores as node-level *data*.
   * - ``template<G,O,X> void add_edge_truth_feature(X fx, std::string name)``
     - Evaluates *fx* on every (topology-filtered) pair ``std::tuple<O*,O*>``
       and stores as edge-level truth feature.
   * - ``template<G,O,X> void add_edge_data_feature(X fx, std::string name)``
     - Like ``add_edge_truth_feature`` but stores as edge-level *data*.
   * - ``template<G> G* get_event()``
     - Returns a typed pointer to the event associated with this graph
       (``static_cast<G*>(m_event)``).
   * - ``bool operator == (graph_template& p)``
     - Hash-equality comparison.
   * - ``void flush_particles()``
     - Frees node-particle resources accumulated during ``CompileEvent``.

Feature Naming Convention
~~~~~~~~~~~~~~~~~~~~~~~~~

Truth features are prefixed ``T-`` and data features ``D-`` internally.  The
*name* argument to the ``add_*`` methods should **not** include this prefix; it
is added automatically.

Example::

    void MyGraph::CompileEvent() {
        MyEvent* ev = this->get_event<MyEvent>();
        std::vector<particle_template*> nodes;
        for (auto& [h, j] : ev->m_jets) {
            nodes.push_back((particle_template*)j);
        }
        this->define_particle_nodes(&nodes);

        // node data: jet pt
        this->add_node_data_feature<double, MyJet>(
            &MyJet::get_pt, "pt"
        );

        // edge truth: same parent top
        this->add_edge_truth_feature<int, MyJet>(
            [](std::tuple<MyJet*, MyJet*>* p) -> int {
                return std::get<0>(*p)->index == std::get<1>(*p)->index;
            }, "same_top"
        );
    }
