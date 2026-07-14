ROC (Python)
============

The ``ROC`` Cython class wraps the C++ ``roc`` curve engine.  It inherits
from :class:`TLine` (via the C++ ``roc → plotting`` chain) so all
:class:`BasePlotting` properties are available.

Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``Scores``
     - ``list[float]`` *(write-only)*
     - List of classifier score values (one per event).  Triggers
       internal ROC computation on assignment.
   * - ``Truth``
     - ``list[int]`` *(write-only)*
     - Ground-truth binary labels (0 or 1, one per event).
   * - ``xBins``
     - ``int``
     - Number of threshold steps used to build the ROC curve (default
       ``100``).
   * - ``AUC``
     - ``float`` *(read-only)*
     - Area Under the ROC Curve, computed after setting ``Scores`` and
       ``Truth``.
   * - ``Titles``
     - ``list[str]``
     - Legend labels for each ROC curve when plotting multiple classifiers.
   * - ``xData``
     - ``list[float]`` *(write-only, blocked)*
     - Blocked — use ``Scores`` instead.  Setting raises a warning.
   * - ``yData``
     - ``list[float]`` *(write-only, blocked)*
     - Blocked — use ``Truth`` instead.  Setting raises a warning.

Usage Example
-------------

.. code-block:: python

   from AnalysisG.core.roc import ROC

   r = ROC()
   r.Scores = scores.tolist()
   r.Truth  = truth.tolist()
   r.xBins  = 200
   r.Filename = "roc_curve"
   r.OutputDirectory = "plots/"
   r.SaveFigure()
   print("AUC:", r.AUC)
