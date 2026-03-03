ModelTemplate (Python)
======================

The ``ModelTemplate`` Cython class wraps the C++ ``model_template``.
User GNN model classes must subclass both ``ModelTemplate`` and
``torch.nn.Module``, then override ``forward(data: graph_t*)``.

Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``o_graph``
     - ``dict[str, str]``
     - Output graph-level feature map (feature name → output key).
   * - ``o_node``
     - ``dict[str, str]``
     - Output node-level feature map.
   * - ``o_edge``
     - ``dict[str, str]``
     - Output edge-level feature map.
   * - ``i_graph``
     - ``list[str]``
     - Input graph-level feature names expected by the model.
   * - ``i_node``
     - ``list[str]``
     - Input node-level feature names expected by the model.
   * - ``i_edge``
     - ``list[str]``
     - Input edge-level feature names expected by the model.
   * - ``device``
     - ``str``
     - Target compute device (e.g. ``"cpu"``, ``"cuda:0"``).
   * - ``checkpoint_path``
     - ``str``
     - Directory where model checkpoints are written/read.
   * - ``weight_name``
     - ``str``
     - File stem used when saving model weights.
   * - ``tree_name``
     - ``str``
     - ROOT tree name used when saving predictions.
   * - ``name``
     - ``str``
     - Model name string.

C++ Interface (called from ``forward``)
----------------------------------------

These methods are available on the ``graph_t*`` pointer passed to ``forward``:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Method
     - Description
   * - ``data.get_data_graph(name: bytes, model: self) → torch.Tensor``
     - Retrieve a graph-level data tensor by name.
   * - ``data.get_data_node(name: bytes, model: self) → torch.Tensor``
     - Retrieve a node-level data tensor by name.
   * - ``data.get_data_edge(name: bytes, model: self) → torch.Tensor``
     - Retrieve an edge-level data tensor by name.
   * - ``data.get_truth_graph(name: bytes, model: self) → torch.Tensor``
     - Retrieve a graph-level truth tensor by name.
   * - ``data.get_truth_node(name: bytes, model: self) → torch.Tensor``
     - Retrieve a node-level truth tensor by name.
   * - ``data.get_truth_edge(name: bytes, model: self) → torch.Tensor``
     - Retrieve an edge-level truth tensor by name.
   * - ``data.get_edge_index(model: self) → torch.Tensor``
     - Retrieve the ``[2, E]`` edge-index tensor.
   * - ``prediction_graph_feature(name: bytes, tensor)``
     - Store a graph-level prediction tensor.
   * - ``prediction_node_feature(name: bytes, tensor)``
     - Store a node-level prediction tensor.
   * - ``prediction_edge_feature(name: bytes, tensor)``
     - Store an edge-level prediction tensor.
