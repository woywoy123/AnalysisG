Grift Model (``models.grift``)
==============================

The ``Grift`` Graph Neural Network is the primary model shipped with
AnalysisG.  Import with::

    from AnalysisG.models.grift import Grift

Grift
-----

``Grift`` is a :class:`~AnalysisG.core.model_template.ModelTemplate`
subclass wrapping the ``grift`` C++ class (``<models/grift.h>``).

Architecture
^^^^^^^^^^^^

Grift is a message-passing GNN with an optional PageRank readout that
classifies edge connectivity (top-quark assignment) and event-level
properties simultaneously.

Hyper-parameters
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``xrec``
     - ``int``
     - Number of recurrent message-passing steps.  Default ``1``.
   * - ``x``
     - ``int``
     - Hidden-layer width (feature dimensionality).  Default ``64``.
   * - ``drop_out``
     - ``float``
     - Dropout probability applied to hidden layers.  Default ``0.0``.
   * - ``is_mc``
     - ``bool``
     - When ``True`` MC-truth features (e.g. ``from_res``) are added to
       the input tensor.  Default ``True``.
   * - ``PageRank``
     - ``bool``
     - When ``True`` a PageRank score is computed for each node during
       inference and stored in the output node tensor.  Default ``False``.

Usage
^^^^^

.. code-block:: python

    from AnalysisG import Analysis
    from AnalysisG.models.grift import Grift

    ana = Analysis()
    ana.AddModel(Grift)
    ana.Epochs = 20
    ana.Start()
