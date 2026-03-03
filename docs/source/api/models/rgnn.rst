Recursive Graph Neural Network (``models.RecursiveGraphNeuralNetwork``)
========================================================================

Import with::

    from AnalysisG.models.RecursiveGraphNeuralNetwork import RecursiveGraphNeuralNetwork

RecursiveGraphNeuralNetwork
---------------------------

``RecursiveGraphNeuralNetwork`` is a
:class:`~AnalysisG.core.model_template.ModelTemplate` subclass wrapping
the ``recursivegraphneuralnetwork`` C++ class
(``<models/RecursiveGraphNeuralNetwork.h>``).

The model performs recursive graph convolutions over node/edge features
to reconstruct top-quark decay topologies.

Hyper-parameters
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``dx``
     - ``int``
     - Input feature width for each node.  Default ``1024``.
   * - ``x``
     - ``int``
     - Hidden-layer feature width.  Default ``64``.
   * - ``output``
     - ``int``
     - Number of output classes.  Default ``1``.
   * - ``rep``
     - ``int``
     - Number of recursive message-passing repetitions.  Default ``1024``.
   * - ``drop_out``
     - ``float``
     - Dropout probability.  Default ``0.1``.
   * - ``res_mass``
     - ``float``
     - Target resonance mass [GeV] used as an auxiliary loss constraint.
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
