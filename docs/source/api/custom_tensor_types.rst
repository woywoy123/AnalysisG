.. _custom-tensor-types:

Adding New Tensor Types
=======================

When defining Graph Neural Networks or other ML representations, you may want to introduce custom features (Node, Edge, or Graph-level tensors). 
In AnalysisG, this is handled gracefully through the ``graph_template`` interface.

Adding Features in ``graph_template``
-------------------------------------
The ``graph_template.h`` and its C++ implementations provide high-level abstractions to construct PyTorch tensors (``torch::Tensor``) from event data. Features are grouped into three categories:

1. **Graph Features**: Overall properties (e.g., missing energy, event weights).
2. **Node Features**: Particle-level properties (e.g., $p_T$, $\eta$, $\phi$).
3. **Edge Features**: Interaction or topological properties (e.g., $\Delta R$, combinations of node states).

Using the Template Methods
--------------------------
To define a new tensor type or feature, invoke the appropriate ``add_`` method inside your event/graph compilation loop.

For Truth-level features:

* ``add_graph_truth_feature(event_object, lambda_getter, "feature_name")``
* ``add_node_truth_feature(lambda_getter, "feature_name")``
* ``add_edge_truth_feature(lambda_getter, "feature_name")``

For Data/Reconstruction-level features:

* ``add_graph_data_feature(event_object, lambda_getter, "feature_name")``
* ``add_node_data_feature(lambda_getter, "feature_name")``
* ``add_edge_data_feature(lambda_getter, "feature_name")``

Under the hood, these templates collect the data (e.g., iterating through all initialized nodes or edges) and pass it to the underlying type-specific handlers (like ``add_node_feature(std::vector<float>, std::string)``), which serialize it into a ``torch::Tensor``.

Topology & Edges
----------------
Before extracting edge features, the topology of the graph must be defined. By default, ``define_topology()`` can be called with a lambda function that evaluates pairs of particles to decide if an edge exists, generating a fully connected graph or specific pruned architectures.
