Recursive Graph Neural Network (``models.RecursiveGraphNeuralNetwork``)
========================================================================

Import with::

    from AnalysisG.models.RecursiveGraphNeuralNetwork import RecursiveGraphNeuralNetwork

RecursiveGraphNeuralNetwork
---------------------------

``RecursiveGraphNeuralNetwork`` is a
:class:`~AnalysisG.core.model_template.ModelTemplate` subclass wrapping
the ``recursivegraphneuralnetwork`` C++ class.

The model performs **recursive message-passing** over node and edge features
using 8 sub-networks to simultaneously predict edge-level top assignments,
node-level aggregated features, and event-level exotic resonance and
number-of-tops outputs.

Architecture (from ``RecursiveGraphNeuralNetwork.cxx``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Constructor signature: ``recursivegraphneuralnetwork(int rep=1024, double drop_out=0.1)``

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Sub-network
     - Role
   * - ``rnn_dx``
     - Edge message network: ``(_dx + 2×_rep) → (_rep×2) → _rep``; LayerNorm + SiLU + Dropout + Sigmoid
   * - ``rnn_x``
     - Node encoder: ``(_x + _rep) → (_rep×2) → _rep``; LayerNorm + Dropout
   * - ``rnn_merge``
     - Hidden-state merge: ``(_rep×3) → _rep``; LayerNorm + SiLU + Dropout
   * - ``rnn_update``
     - Edge-prediction updater: ``(_output×2 + _rep×2) → _output``; LayerNorm + SiLU + Dropout
   * - ``exotic_mlp``
     - Exotic resonance head: ``(_x + _output) → (_rep×2) → _output``; LayerNorm + Dropout
   * - ``node_aggr_mlp``
     - Node aggregation: ``_x → _rep → _x``; LayerNorm + Dropout
   * - ``ntops_mlp``
     - Number-of-tops head: ``(_x + 4) → _rep → _x``; LayerNorm + SiLU + ReLU
   * - ``exo_mlp``
     - Second exotic head (post-aggregation): same pattern as ``ntops_mlp``

Architecture dimensions:

.. list-table::
   :header-rows: 1
   :widths: 15 10 75

   * - Field
     - Default
     - Meaning
   * - ``_dx``
     - 26
     - Input edge-feature dimension (concatenated node features for both endpoints)
   * - ``_x``
     - 5
     - Input node-feature dimension
   * - ``_output``
     - 2
     - Output edge-prediction dimension (binary: same-top or not)
   * - ``_rep``
     - 256
     - Internal hidden-state dimension (overridden by constructor argument)

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
     - Dropout probability.  Default ``0.1``.
   * - ``res_mass``
     - ``float``
     - Target resonance mass [MeV] used as an auxiliary loss constraint.
       Default ``0.0`` (disabled).
   * - ``is_mc``
     - ``bool``
     - Include MC-truth features in the input.  Default ``True``.

Usage
^^^^^

.. code-block:: python

    from AnalysisG import Analysis
    from AnalysisG.models.RecursiveGraphNeuralNetwork import RecursiveGraphNeuralNetwork

    ana = Analysis()
    ana.AddModel(RecursiveGraphNeuralNetwork)
    ana.Epochs = 20
    ana.Start()

C++ Reference
^^^^^^^^^^^^^

.. doxygenclass:: recursivegraphneuralnetwork
   :project: AnalysisG
   :members:
   :undoc-members:
