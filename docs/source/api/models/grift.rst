Grift Model (``models.grift``)
==============================

The ``Grift`` Graph Neural Network is a message-passing GNN for
top-quark combinatorics.  Import with::

    from AnalysisG.models.grift import Grift

Grift
-----

``Grift`` is a :class:`~AnalysisG.core.model_template.ModelTemplate`
subclass wrapping the ``grift`` C++ class.

Architecture (from ``grift.cxx``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Grift uses **7 torch.nn.Sequential sub-networks** arranged in a recurrent
message-passing loop:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Sub-network
     - Role
   * - ``rnn_x``
     - Node encoder: ``(_xin + _xrec) → (_xrec + _xin) → _xrec``; uses LayerNorm + Tanh + LeakyReLU
   * - ``rnn_dx``
     - Edge message network: ``(_xrec × 3) → (_xrec × 2) → _xrec``; Tanh activations
   * - ``rnn_hxx``
     - Hidden-state aggregator: ``(_xrec×2 + 2) → (_xrec×2) → _xrec``; LeakyReLU + Tanh
   * - ``rnn_txx``
     - Top-edge predictor: ``(_xrec × 3) → (_xrec × 2) → _xout``; LeakyReLU + Tanh
   * - ``rnn_rxx``
     - Resonance predictor: ``(_xrec×4) → _hidden → _xout``; LeakyReLU + Sigmoid
   * - ``mlp_ntop``
     - Number-of-tops readout: ``(_xtop + _xrec) → _xrec → _xtop``; LayerNorm + LeakyReLU + Sigmoid
   * - ``mlp_sig``
     - Signal classifier: ``(_xout×2 + _xrec×4 + _xtop×2) → (_xrec×2) → _xout``; LayerNorm + LeakyReLU + Sigmoid

Architecture dimensions (C++ defaults):

.. list-table::
   :header-rows: 1
   :widths: 15 10 75

   * - Field
     - Default
     - Meaning
   * - ``_hidden``
     - 1024
     - Width of the ``rnn_rxx`` hidden layer
   * - ``_xrec``
     - 128
     - Recurrent hidden-state dimension used by all sub-networks
   * - ``_xin``
     - 6
     - Input node-feature dimension (``pt, eta, phi, E, is_lep, is_b``)
   * - ``_xout``
     - 2
     - Output edge/graph prediction dimension (binary classification)
   * - ``_xtop``
     - 5
     - Number-of-tops feature dimension

Hyper-parameters (Python layer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``drop_out``
     - ``float``
     - Dropout probability applied to hidden layers.  Default ``0.0``.
   * - ``is_mc``
     - ``bool``
     - When ``True`` MC-truth features are included in the input tensor.  Default ``True``.
   * - ``pagerank``
     - ``bool``
     - When ``True`` a PageRank score is computed for each node during
       inference and stored in the output node tensor.  Default ``False``.

Expected Input Features
^^^^^^^^^^^^^^^^^^^^^^^

Grift reads the following tensors from ``graph_t`` inside ``forward()``:

.. list-table::
   :header-rows: 1

   * - Feature
     - Slot
     - Registered by
   * - ``"pt"``
     - node data
     - graph class
   * - ``"eta"``
     - node data
     - graph class
   * - ``"phi"``
     - node data
     - graph class
   * - ``"energy"``
     - node data
     - graph class
   * - ``"is_lep"``
     - node data
     - graph class
   * - ``"is_b"``
     - node data
     - graph class
   * - edge index
     - COO
     - graph class

Usage
^^^^^

.. code-block:: python

    from AnalysisG import Analysis
    from AnalysisG.models.grift import Grift

    ana = Analysis()
    ana.AddModel(Grift)
    ana.Epochs = 20
    ana.Start()

C++ Reference
^^^^^^^^^^^^^

.. doxygenclass:: grift
   :project: AnalysisG
   :members:
   :undoc-members:
